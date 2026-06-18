# Changes

## 1.2.0 (2026-06-18)

- Migrate from Poetry to uv for package management
- Replace `poetry.lock` with `uv.lock`
- Convert dependencies to PEP 508 format, switch build backend to hatchling
- Update CI workflows to use `astral-sh/setup-uv@v5`
- Update Makefile targets to use `uv sync` / `uv run` / `uv build`
- `--file` now loads TART .h5 visibility files (calibrated visibilities from telescope)
- Remove broken `--api` fallback: `--file` or `--ms` is now required
- Remove unused imports (`json`, `deepcopy`, `settings`, `api_imaging`)
