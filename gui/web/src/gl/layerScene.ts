/**
 * A small instanced-box renderer for the model layer map.
 *
 * Hand-written rather than pulled from a 3D library, for two reasons. The scene
 * is one primitive drawn a few hundred times, which is a few hundred lines of
 * WebGL and about half a megabyte of dependency. And the colours have to come
 * from the app's CSS tokens so the 3D view matches the DOM around it in both
 * themes -- easy when the shader is ours, awkward when a library owns the
 * material system.
 *
 * Picking is done on the CPU by intersecting a ray with each instance's box.
 * The usual alternative is a second render pass to an id buffer; with a few
 * hundred boxes the arithmetic is far cheaper than the extra pass and the
 * readback stall that comes with it.
 */

/** One box in the scene. Positions and sizes are in scene units, not pixels. */
export interface LayerInstance {
  /** Stable identifier handed back by pick(); the caller decides what it means. */
  id: string
  /** Centre of the box. */
  x: number
  y: number
  z: number
  /** Full extents, not half-extents. */
  sx: number
  sy: number
  sz: number
  /** Linear 0..1 colour, already resolved from theme tokens by the caller. */
  r: number
  g: number
  b: number
  /** Opacity; used to recede blocks outside the current selection. */
  a: number
}

export interface SceneTheme {
  /** Cleared background. Matches the surrounding panel so the canvas is not a hole. */
  background: [number, number, number]
  /** Outline colour for the selected instances. */
  highlight: [number, number, number]
}

export interface OrbitState {
  /** Horizontal angle, radians. */
  azimuth: number
  /** Vertical angle, radians, clamped away from the poles. */
  elevation: number
  /** Distance from the target. */
  distance: number
}

const VERTEX_SHADER = `#version 300 es
precision highp float;

// Unit cube, centred on the origin.
in vec3 a_position;
in vec3 a_normal;

// Per instance.
in vec3 i_offset;
in vec3 i_scale;
in vec4 i_color;

uniform mat4 u_viewProjection;

out vec3 v_normal;
out vec4 v_color;
out vec3 v_local;

void main() {
  vec3 world = a_position * i_scale + i_offset;
  gl_Position = u_viewProjection * vec4(world, 1.0);
  // Non-uniform scale would shear a normal, but every box here is axis-aligned
  // so dividing by the scale is enough to keep it correct.
  v_normal = normalize(a_normal / max(i_scale, vec3(0.0001)));
  v_color = i_color;
  v_local = a_position;
}
`

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec3 v_normal;
in vec4 v_color;
in vec3 v_local;

uniform vec3 u_lightDirection;

out vec4 outColor;

void main() {
  // Two-term lighting: a key light plus a constant ambient floor, so a face
  // turned away from the light is still readable rather than black. A third
  // fill term from below keeps the underside of the stack from going flat.
  float key = max(dot(v_normal, u_lightDirection), 0.0);
  float fill = max(dot(v_normal, vec3(0.0, -1.0, 0.0)), 0.0) * 0.15;
  float ambient = 0.45;
  float lighting = ambient + key * 0.55 + fill;

  // A faint darkening towards each face's edge reads as a seam between
  // neighbouring boxes without needing a wireframe pass.
  vec3 edge = abs(v_local) * 2.0;
  float nearEdge = max(max(edge.x, edge.y), edge.z);
  float seam = 1.0 - smoothstep(0.90, 1.0, nearEdge) * 0.25;

  outColor = vec4(v_color.rgb * lighting * seam, v_color.a);
}
`

/* --------------------------------------------------------------------------
   Minimal 4x4 matrix maths. Column-major, matching WebGL's expectations.
   -------------------------------------------------------------------------- */

function perspective(fovRadians: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovRadians / 2)
  const range = 1 / (near - far)
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * range, -1,
    0, 0, near * far * range * 2, 0,
  ])
}

function lookAt(eye: number[], target: number[], up: number[]): Float32Array {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]])
  const x = normalize(cross(up, z))
  const y = cross(z, x)
  // prettier-ignore
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ])
}

function multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16)
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      let sum = 0
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + row] * b[column * 4 + k]
      out[column * 4 + row] = sum
    }
  }
  return out
}

const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: number[], b: number[]) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
function normalize(v: number[]): number[] {
  const length = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / length, v[1] / length, v[2] / length]
}

/* --------------------------------------------------------------------------
   Cube geometry: 6 faces, 2 triangles each, with a per-face normal.
   -------------------------------------------------------------------------- */

function buildCube(): { positions: Float32Array; normals: Float32Array } {
  const faces: { normal: number[]; corners: number[][] }[] = [
    { normal: [0, 0, 1], corners: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
    { normal: [0, 0, -1], corners: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]] },
    { normal: [1, 0, 0], corners: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]] },
    { normal: [-1, 0, 0], corners: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]] },
    { normal: [0, 1, 0], corners: [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]] },
    { normal: [0, -1, 0], corners: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]] },
  ]
  const positions: number[] = []
  const normals: number[] = []
  for (const face of faces) {
    const [a, b, c, d] = face.corners
    for (const corner of [a, b, c, a, c, d]) {
      positions.push(corner[0], corner[1], corner[2])
      normals.push(face.normal[0], face.normal[1], face.normal[2])
    }
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals) }
}

/* -------------------------------------------------------------------------- */

export interface LayerScene {
  setInstances(instances: LayerInstance[]): void
  setTheme(theme: SceneTheme): void
  setHighlight(ids: Set<string>): void
  getOrbit(): OrbitState
  setOrbit(next: Partial<OrbitState>): void
  resize(): void
  render(): void
  /** Scene id under the given canvas-relative pixel, or null. */
  pick(pixelX: number, pixelY: number): string | null
  /**
   * World point -> CSS pixel position within the canvas, or null when the point
   * is behind the camera. Used to hang HTML labels on the scene: text drawn in
   * WebGL would need a glyph atlas and would not inherit the app's font,
   * whereas a positioned DOM element is themed, selectable, and readable by a
   * screen reader for free.
   */
  project(x: number, y: number, z: number): { x: number; y: number } | null
  dispose(): void
}

/**
 * Create the renderer, or return null when WebGL2 is unavailable.
 *
 * Returning null rather than throwing is deliberate: the pane has a complete
 * non-visual path (the layer table is the real selection UI), so a machine
 * without WebGL loses the picture and keeps the function.
 */
export function createLayerScene(canvas: HTMLCanvasElement): LayerScene | null {
  const gl = canvas.getContext('webgl2', {
    antialias: true,
    alpha: false,
    // The scene is redrawn on demand, not every frame, so the buffer has to
    // survive between draws.
    preserveDrawingBuffer: true,
  })
  if (!gl) return null

  const compile = (type: number, source: string) => {
    const shader = gl.createShader(type)
    if (!shader) throw new Error('Could not create shader')
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader)
      gl.deleteShader(shader)
      throw new Error(`Shader compile failed: ${log}`)
    }
    return shader
  }

  let program: WebGLProgram
  try {
    const vertex = compile(gl.VERTEX_SHADER, VERTEX_SHADER)
    const fragment = compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER)
    const created = gl.createProgram()
    if (!created) throw new Error('Could not create program')
    gl.attachShader(created, vertex)
    gl.attachShader(created, fragment)
    gl.linkProgram(created)
    if (!gl.getProgramParameter(created, gl.LINK_STATUS)) {
      throw new Error(`Program link failed: ${gl.getProgramInfoLog(created)}`)
    }
    gl.deleteShader(vertex)
    gl.deleteShader(fragment)
    program = created
  } catch {
    return null
  }

  const cube = buildCube()
  const vao = gl.createVertexArray()
  gl.bindVertexArray(vao)

  const positionBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, cube.positions, gl.STATIC_DRAW)
  const positionLocation = gl.getAttribLocation(program, 'a_position')
  gl.enableVertexAttribArray(positionLocation)
  gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0)

  const normalBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, cube.normals, gl.STATIC_DRAW)
  const normalLocation = gl.getAttribLocation(program, 'a_normal')
  gl.enableVertexAttribArray(normalLocation)
  gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, 0, 0)

  // One interleaved buffer for all per-instance data: offset(3) scale(3) color(4).
  const INSTANCE_FLOATS = 10
  const instanceBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer)
  const stride = INSTANCE_FLOATS * 4
  const bindInstanceAttribute = (name: string, size: number, offsetFloats: number) => {
    const location = gl.getAttribLocation(program, name)
    if (location < 0) return
    gl.enableVertexAttribArray(location)
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offsetFloats * 4)
    gl.vertexAttribDivisor(location, 1)
  }
  bindInstanceAttribute('i_offset', 3, 0)
  bindInstanceAttribute('i_scale', 3, 3)
  bindInstanceAttribute('i_color', 4, 6)
  gl.bindVertexArray(null)

  const viewProjectionLocation = gl.getUniformLocation(program, 'u_viewProjection')
  const lightLocation = gl.getUniformLocation(program, 'u_lightDirection')

  let instances: LayerInstance[] = []
  let instanceData = new Float32Array(0)
  let highlight = new Set<string>()
  let theme: SceneTheme = { background: [0.1, 0.1, 0.1], highlight: [1, 1, 1] }
  const orbit: OrbitState = { azimuth: -0.42, elevation: 0.34, distance: 34 }
  // The stack is built upward from y=0, so the camera looks at its middle.
  let target: [number, number, number] = [0, 0, 0]

  const uploadInstances = () => {
    instanceData = new Float32Array(instances.length * INSTANCE_FLOATS)
    instances.forEach((instance, index) => {
      const base = index * INSTANCE_FLOATS
      instanceData[base + 0] = instance.x
      instanceData[base + 1] = instance.y
      instanceData[base + 2] = instance.z
      instanceData[base + 3] = instance.sx
      instanceData[base + 4] = instance.sy
      instanceData[base + 5] = instance.sz
      const selected = highlight.size === 0 || highlight.has(instance.id)
      // Unselected blocks recede rather than disappear: the shape of the whole
      // model stays legible while the chosen range reads as the subject.
      const lift = selected ? 1 : 0.55
      instanceData[base + 6] = instance.r * lift + (selected ? 0 : 0.04)
      instanceData[base + 7] = instance.g * lift + (selected ? 0 : 0.04)
      instanceData[base + 8] = instance.b * lift + (selected ? 0 : 0.04)
      instanceData[base + 9] = selected ? instance.a : instance.a * 0.75
    })
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, instanceData, gl.DYNAMIC_DRAW)
  }

  const cameraBasis = () => {
    const { azimuth, elevation, distance } = orbit
    const eye: [number, number, number] = [
      target[0] + distance * Math.cos(elevation) * Math.sin(azimuth),
      target[1] + distance * Math.sin(elevation),
      target[2] + distance * Math.cos(elevation) * Math.cos(azimuth),
    ]
    const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]])
    const right = normalize(cross(forward, [0, 1, 0]))
    const up = cross(right, forward)
    return { eye, forward, right, up }
  }

  const FOV = (45 * Math.PI) / 180

  /** The eight corners of the content's bounding box, in world space. */
  let contentCorners: number[][] = []

  /**
   * Distance at which the content just fits the frame at the current angle.
   *
   * A bounding sphere would be simpler and orientation-independent, but this
   * content is long and thin: a sphere around a 28-layer stack has a radius set
   * by its diagonal, so fitting it leaves the model small in the middle of a
   * large empty frame. Measuring the corners in the camera's own right/up basis
   * fits the actual silhouette instead.
   *
   * Called when the content or the aspect changes, not while orbiting -- a
   * viewer that re-zoomed continuously during a drag is disorienting to use.
   */
  const fitDistance = () => {
    if (contentCorners.length === 0) return orbit.distance
    const aspect = canvas.width / Math.max(1, canvas.height)
    const tanVertical = Math.tan(FOV / 2)
    const tanHorizontal = tanVertical * aspect
    const { forward, right, up } = cameraBasis()

    let halfWidth = 0
    let halfHeight = 0
    let towardCamera = 0
    for (const corner of contentCorners) {
      const rel = [corner[0] - target[0], corner[1] - target[1], corner[2] - target[2]]
      halfWidth = Math.max(halfWidth, Math.abs(dot(rel, right)))
      halfHeight = Math.max(halfHeight, Math.abs(dot(rel, up)))
      // Negative along forward is toward the eye; that part of the model needs
      // extra clearance or the nearest corner pokes through the near plane.
      towardCamera = Math.max(towardCamera, -dot(rel, forward))
    }

    const needed = Math.max(halfWidth / tanHorizontal, halfHeight / tanVertical)
    // The 1.08 is breathing room so the model does not touch the frame edge.
    return Math.max(4, (needed + towardCamera) * 1.08)
  }

  const render = () => {
    const width = canvas.width
    const height = canvas.height
    if (width === 0 || height === 0) return

    gl.viewport(0, 0, width, height)
    gl.clearColor(theme.background[0], theme.background[1], theme.background[2], 1)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    if (instances.length === 0) return

    gl.enable(gl.DEPTH_TEST)
    gl.enable(gl.CULL_FACE)
    gl.cullFace(gl.BACK)

    const { eye } = cameraBasis()
    const view = lookAt(eye as number[], target as number[], [0, 1, 0])
    const projection = perspective(FOV, width / Math.max(1, height), 0.1, 500)

    gl.useProgram(program)
    gl.uniformMatrix4fv(viewProjectionLocation, false, multiply(projection, view))
    // A fixed light in view-independent world space: rotating the model changes
    // which faces are lit, which is most of what sells the shape as solid.
    gl.uniform3fv(lightLocation, normalize([0.45, 0.8, 0.4]))

    gl.bindVertexArray(vao)
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, instances.length)
    gl.bindVertexArray(null)
  }

  return {
    setInstances(next) {
      instances = next
      if (next.length > 0) {
        // Measure the content, including the Z extrusion the metric drives,
        // then hand the corners to fitDistance for framing at the current
        // viewing angle.
        let minX = Infinity, maxX = -Infinity
        let minY = Infinity, maxY = -Infinity
        let minZ = Infinity, maxZ = -Infinity
        for (const instance of next) {
          minX = Math.min(minX, instance.x - instance.sx / 2)
          maxX = Math.max(maxX, instance.x + instance.sx / 2)
          minY = Math.min(minY, instance.y - instance.sy / 2)
          maxY = Math.max(maxY, instance.y + instance.sy / 2)
          minZ = Math.min(minZ, instance.z - instance.sz / 2)
          maxZ = Math.max(maxZ, instance.z + instance.sz / 2)
        }
        target = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2]
        contentCorners = []
        for (const x of [minX, maxX]) {
          for (const y of [minY, maxY]) {
            for (const z of [minZ, maxZ]) contentCorners.push([x, y, z])
          }
        }
        orbit.distance = fitDistance()
      }
      uploadInstances()
      render()
    },

    setTheme(next) {
      theme = next
      render()
    },

    setHighlight(ids) {
      highlight = ids
      uploadInstances()
      render()
    },

    getOrbit: () => ({ ...orbit }),

    setOrbit(next) {
      if (next.azimuth !== undefined) orbit.azimuth = next.azimuth
      if (next.elevation !== undefined) {
        // Stop just short of the poles; at exactly vertical the up vector and
        // the view direction become parallel and the basis collapses.
        orbit.elevation = Math.max(-1.45, Math.min(1.45, next.elevation))
      }
      if (next.distance !== undefined) {
        orbit.distance = Math.max(4, Math.min(400, next.distance))
      }
      render()
    },

    resize() {
      const rect = canvas.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      const width = Math.max(1, Math.round(rect.width * ratio))
      const height = Math.max(1, Math.round(rect.height * ratio))
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        // Aspect changed, so the distance that just fitted no longer does.
        orbit.distance = fitDistance()
      }
      render()
    },

    render,

    pick(pixelX, pixelY) {
      const rect = canvas.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0 || instances.length === 0) return null

      // Canvas pixel -> normalised device coordinates -> a world-space ray.
      // Building the ray from the camera basis avoids inverting the matrix.
      const ndcX = (pixelX / rect.width) * 2 - 1
      const ndcY = 1 - (pixelY / rect.height) * 2
      const aspect = rect.width / rect.height
      const tangent = Math.tan(FOV / 2)
      const { eye, forward, right, up } = cameraBasis()
      const direction = normalize([
        forward[0] + right[0] * ndcX * tangent * aspect + up[0] * ndcY * tangent,
        forward[1] + right[1] * ndcX * tangent * aspect + up[1] * ndcY * tangent,
        forward[2] + right[2] * ndcX * tangent * aspect + up[2] * ndcY * tangent,
      ])

      let nearest: string | null = null
      let nearestDistance = Infinity
      for (const instance of instances) {
        // Slab test: clip the ray against each axis pair in turn. If the
        // entry point is ever beyond the exit point the ray misses the box.
        let tMin = 0
        let tMax = Infinity
        const centre = [instance.x, instance.y, instance.z]
        const half = [instance.sx / 2, instance.sy / 2, instance.sz / 2]
        let hit = true
        for (let axis = 0; axis < 3; axis += 1) {
          const originOffset = eye[axis] - centre[axis]
          if (Math.abs(direction[axis]) < 1e-8) {
            // Ray is parallel to this slab; it misses unless it starts inside.
            if (Math.abs(originOffset) > half[axis]) {
              hit = false
              break
            }
            continue
          }
          const inverse = 1 / direction[axis]
          let t1 = (-half[axis] - originOffset) * inverse
          let t2 = (half[axis] - originOffset) * inverse
          if (t1 > t2) [t1, t2] = [t2, t1]
          tMin = Math.max(tMin, t1)
          tMax = Math.min(tMax, t2)
          if (tMin > tMax) {
            hit = false
            break
          }
        }
        if (hit && tMin < nearestDistance) {
          nearestDistance = tMin
          nearest = instance.id
        }
      }
      return nearest
    },

    project(x, y, z) {
      const rect = canvas.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return null
      const { eye } = cameraBasis()
      const view = lookAt(eye as number[], target as number[], [0, 1, 0])
      const projection = perspective(FOV, rect.width / rect.height, 0.1, 500)
      const matrix = multiply(projection, view)
      const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]
      const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]
      const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15]
      // A non-positive w means the point is at or behind the eye plane, where
      // the perspective divide would mirror it to the wrong side of the screen.
      if (clipW <= 0) return null
      return {
        x: ((clipX / clipW) * 0.5 + 0.5) * rect.width,
        y: (1 - ((clipY / clipW) * 0.5 + 0.5)) * rect.height,
      }
    },

    dispose() {
      gl.deleteBuffer(positionBuffer)
      gl.deleteBuffer(normalBuffer)
      gl.deleteBuffer(instanceBuffer)
      gl.deleteVertexArray(vao)
      gl.deleteProgram(program)
    },
  }
}
