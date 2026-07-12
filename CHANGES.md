# Changes

## 1.4.3 (2026-06-22)

- Make `gmsh` and `meshio` optional dependencies in new `[mesh]` extras group; `uv sync --extra mesh` to install
- Clear error messages when using `--mesh` without the optional dependencies

## 1.4.2 (2026-06-22)

- Delegate angle parsing to `angle-parser` library; remove custom `parse_ending` and unit constants from `Resolution`
- Add `angle-parser>=0.2.0` dependency; remove fallback for `mas`, `uas` (now handled natively)
- Lower minimum gmsh version from 4.9 to 4.0
- Update optional `tart2ms` dependency from >=0.7.1 to >=0.9.0
- Update `uv.lock` for all dependency changes

## 1.4.1 (2026-06-22)

- Remove redundant `DirectImagingOperator` (10% code reduction); `DiSkOOperator` handles the exact adjoint directly via `A.T`.
- Fix FISTA and LSMR initial guess: Use true adjoint `np.abs(A.T @ d)` to guarantee a physically correct (non-mirrored) starting image.

## 1.4.0 (2026-06-22)

- Convert from pygmsh to gmsh native API; remove pygmsh dependency; add meshio
- Select visibilities from all MS snapshots instead of just snapshot 0
- Add baseline length percentile output (0-100% in 5% steps)
- Fix `ms_helper`: rename misleading `res_arcmin` to `res_deg`, remove dead code
- Fix `image_natural`: broadcast bug in sigma_1 division causing wrong shapes
- Cache harmonic blocks per-block (float32) for matrix-free FISTA; limit to <500 MB
- Fix `test_meshsphere`: gmsh compatibility, split standalone `test_areas`
- Fix `test_telescope_operator`: harmonic normalization for physical pixel areas
- Fix `test_pylops_operator`: replace broken `from_resolution()` calls
- Fix `test_sphere`: incorrect zenith physics replaced with round-trip test
- Fix `test_subsphere`: timezone-naive datetime → UTC-aware
- Configure pytest to ignore `context/` directory
- Fix `scipy.misc` deprecation: use `imageio` instead
- Update copyright notices to 2022-2026 across all files

## 1.3.3 (2026-06-19)

- Vectorize `get_harmonics`, `make_gamma`, and matrix-free operators (`DiSkOOperator`, `DirectImagingOperator`) using broadcasting and blocked BLAS-level operations (10-50x speedup)
- Fix FISTA solver: remove unconditional `eps = 1e-9` override, switch from broken analysis formulation (`SOp=Apre`) to synthesis formulation, use proper initial guess (`abs(Apre @ d)`) instead of zeros
- Fix `make_gamma` to use `np.concatenate` instead of `np.block` (avoids 2x memory copy)
- Fix `image_visibilities` to use vectorized `vis_arr @ gamma` instead of Python loop
- Fix FISTA test: correct solver parameters, relax tolerance from `places=3` to `places=1` (FISTA is a first-order method on an ill-conditioned problem)
- Move historical changelog from README.md code block into CHANGES.md as proper markdown
- CI: fix test matrix (remove Python 3.14, add 3.11) to match `requires-python`
- CI: target `disko/` directory in flake8 to avoid scanning dependencies
- Configure flake8 exclusions for `.venv`, `.git`, `__pycache__` in `setup.cfg`
- Fix various flake8 lint issues (unused imports, variables, long lines)
- Fix `disko_bayes` MS reading: replace broken `DiSkO.from_ms()` with `disko_from_ms()`, add visibility conjugation, geo-location, and file-exists check
- Fix `disko_bayes` output: remove invalid `fov` argument from `to_fits()` and `to_svg()` calls
- Fix publish workflow: use concrete Python version (`3.12` instead of `3.x`), pin `actions/checkout@v4`, fix `--outdir` → `--out-dir`

## 1.2.0 (2026-06-18)

- Migrate from Poetry to uv for package management
- Replace `poetry.lock` with `uv.lock`
- Convert dependencies to PEP 508 format, switch build backend to hatchling
- Update CI workflows to use `astral-sh/setup-uv@v5`
- Update Makefile targets to use `uv sync` / `uv run` / `uv build`
- `--file` now loads TART .h5 visibility files (calibrated visibilities from telescope)
- Remove broken `--api` fallback: `--file` or `--ms` is now required
- Remove unused imports (`json`, `deepcopy`, `settings`, `api_imaging`)

## 1.1.0

- Add `--data-column` command line argument to specify the data column (Default DATA)

## 1.0.8

- Fix bug in `--scale-mad` to handle cases when mad == 0

## 1.0.7

- Introduce `--scale-mad` to `disko-draw` to scale pixels to median absolute deviation

## 1.0.1

- Move to poetry
- Update for numpy > 2.0

## 1.0.0b5

- Fix up the inclusion of non tart stuff

## 1.0.0b4

- Remove tart2ms include

## 1.0.0b3

- Remove dask-ms dependency

## 1.0.0b2

- Make tart dependencies optional so allow direct imaging code

## 1.0.0b1

- Change Sphere to FoV. I.e. HealpixFoV (Field of View)
- Add new SquareFoV class for square images (work in progress)

## 0.9.6b2

- Fix the `--elevation` limit to actually implement this for disko draw

## 0.9.6b1

- Add a minimum elevation to the sphere.el_min_r for setting bounds in imagers
- Explicitly manage the tart2ms logging

## 0.9.5b4

- Add a timestamp to images (or a title if specified) in SVG mode
- Clean up logging so that only happens when `--debug` is present

## 0.9.5b3

- Use gmsh rather than optimesh (WIP)
- Use much faster measurement set reading via `casa_read_ms()` about 200x faster
- `disko_draw` timestamps the image

## 0.9.5b2

- Fix RA direction in generated FITS files (thanks Ben Hugo)

## 0.9.5b1

- Add `--min` and `--max` to `disko_draw` to allow manual setting the range of images

## 0.9.4b6

- Fix bug in drawing PDF

## 0.9.4b5

- Import Resolution in disko to get array beam width
- Fix sphere power

## 0.9.4b4

- Expose parent parsers
- Refer to `min_res()` rather than nside for spheres
- Fix bugs in display of mesh spheres
- Add `disko.fov` namespace
- Serialize to hdf5 files
- New `disko_draw` CLI tool
- Conjugate visibilities from files

## 0.9.4b3

- Move sphere args parser to the sphere object

## 0.9.4b2

- Add helper method to calculate beam size
- Add `area()`, `get_power()` method to sphere
- Add `rms()`, `copy()` methods for rms and deep copying of spheres

## 0.9.4b1

- Use `read_ms` from tart2ms (moved there)

## 0.9.3b6

- Use speed of light from `astropy.constants`
- Add a `--version` option to print the current version and exit

## 0.9.3b5

- Fix bug in the Matrix Free Linear Operator which wasn't conjugated

## 0.9.3b4

- Raise nicer errors when arguments aren't provided

## 0.9.3b2

- Fix indexing error in `read_ms` when the number of visibilities requested exceeded the number available
- Clean up the meshing
- Rework the command line interface; new resolution specification
- Output residuals to the terminal
- Use Natural weighting when reading from measurement sets

## 0.9.3b1

- Add `--h5` option to allow sequential inference from a visibility file

## 0.9.2b1

- No longer require arcmin for construction of spheres

## 0.9.1b1

- Remove constraint that nside is a power of two now that healpy has accepted the pull request
- Add new parameter `l1_ratio`
- Don't scale the alpha parameter
- Allow negative solutions for Tikhonov regression
- Allow full skies using `--nside` option
- Add a colour bar to the SVG output

## 0.9.0b4

- Improve measurement set reading
- Use the mean RMS value for a single noise estimate on visibilities
- Use the correct rank value in overdetermined skies
- Truncate the SVD to keep the condition number of the telescope less than 50

## 0.9.0b3

- Full Bayesian Inference is working
- Fix bug in meshio (after upgrade beyond 4)

## 0.9.0b2

- Add a multivariate gaussian object
- Fix ms_helper
- Handle the case where the rank of the telescope operator is not full

## 0.9.0b1

- Move to a real telescope operator

## 0.8.0b5

- Allow FISTA to calculate its own largest eigenvalue if negative values are passed in

## 0.8.0b4

- Clean up code and avoid recalculating harmonics
- Added a `DirectImagingOperator` that performs the discrete Fourier Transform

## 0.8.0b3

- Add `--fista` command line option to use the FISTA solver

## 0.8.0b2

- Add an lsqr option to force the slightly slower lsqr algorithm in place of lsmr

## 0.8.0b1

- Add a matrix-free operator that actually works; process UVW in meters

## 0.7.0b10

- Clean up tests
- Rename the DiSkOOperator and get it going
- Fix up timestamp loading, use the correct frequency (based on channel parameter)

## 0.7.0b9

- Fix up timestamp loading

## 0.7.0b8

- Optimize mesh at each stage of refinement

## 0.7.0b7

- Better refinement

## 0.7.0b6

- Limit gradient calculation to cells above nyquist limit

## 0.7.0b5

- Improve channel selection

## 0.7.0b4

- Allow selection of the channel number

## 0.7.0b2

- New adaptive meshing on gradient

## 0.7.0b1

- Add adaptive meshing and `--adaptive` option

## 0.6.0b9

- Report Nyquist resolution

## 0.6.0b7

- MS were being read incorrectly - the UVW are measured in meters, not wavelengths

## 0.6.0b6

- Correct field pointing from measurement sets

## 0.6.0b5

- Reduce memory requirements by around 25%

## 0.6.0b4

- Report the r^2 value

## 0.6.0b2

- Use dask for very large jobs (use the `--dask` switch)

## 0.6.0b1

- Get data from Measurement Sets

## 0.5.0b5

- Allow sources not to be shown

## 0.5.0b4

- Override plot in HPSubSphere to allow for non-normal pixels

## 0.5.0b3

- Added elliptical source circle projections in SVG

## 0.5.0

- Getting imaging logic better
- Added L2 regularization, and cross-validation
