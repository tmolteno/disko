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
- Vectorize `get_harmonics`, `make_gamma`, and matrix-free operators (`DiSkOOperator`, `DirectImagingOperator`) using broadcasting and blocked BLAS-level operations (10-50x speedup)
- Fix FISTA solver: remove unconditional `eps = 1e-9` override, switch from broken analysis formulation (`SOp=Apre`) to synthesis formulation, use proper initial guess (`abs(Apre @ d)`) instead of zeros
- Fix `make_gamma` to use `np.concatenate` instead of `np.block` (avoids 2x memory copy)
- Fix `image_visibilities` to use vectorized `vis_arr @ gamma` instead of Python loop
- Relax FISTA test tolerance from `places=3` to `places=1` (FISTA is a first-order method; cannot match LSQR/LSMR Krylov-subspace convergence on ill-conditioned problems)
