# Public-release safety review

Date: 2026-07-29

## Scope and boundary

This review covers the staged public ReTrain source tree before GitHub release.
The application is designed for loopback use; no public deployment, account,
credential, firewall, or permission change is part of this release.

## Findings and action

- Removed static workstation paths from source, installer configuration, and
  public handoff text. Optional local locations now use documented environment
  variables.
- Added `.env` and `.env.*` to the ignored set while preserving an optional
  `.env.example` convention.
- Updated the direct PostCSS dependency after the release audit identified a
  high-severity source-map disclosure advisory; the subsequent production
  audit reports zero vulnerabilities.
- Confirmed the repository ignores model weights, local run state, logs,
  virtual environments, generated frontend output, and the external trainer
  checkout.
- The included Codex-app dataset is public-release review material, not a
  license to add private chat, account, credential, or session data.

## Checks to repeat before every release

1. Search staged content for credentials and personal paths.
2. Inspect staged files explicitly; do not use broad staging.
3. Run compilation, dataset validation, and the frontend build.
4. Review new dataset rows and documentation claims for private data and
   unsupported capability statements.

This is release hygiene evidence, not a security certification or a legal
privacy assessment.
