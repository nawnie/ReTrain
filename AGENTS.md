# Rnv1-ReTrain agent instructions

Start by reading `HANDOFF.md`. It has the current state of the ReTrain move,
the dashboard work in progress, and the verification steps that still need to
run.

## Training ownership

- This folder is now the standalone ReTrain workspace for training image, video,
  audio, and chat models.
- AIWF Studio should stay focused on inference and generation workflows.
- Do not wire this folder back into AIWF Studio unless Shawn explicitly asks for integration work.
- Do not treat this as part of MoK unless Shawn explicitly asks for MoK
  integration.

## Current app direction

- FastAPI should be the backend for the ReTrain app.
- The main screen should show user-friendly training metrics derived from
  TensorBoard data.
- Full TensorBoard should be available from a button as an in-app popup, not as
  the default main display and not as a separate app Shawn has to manage.
