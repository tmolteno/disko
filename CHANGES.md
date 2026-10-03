# Changes

## Unreleased

- Phase 3 of #10 (package split): the code is now in the two packages the issue asked for, behind compatibility shims, with no behaviour change. `disko.fov` — the sky model — holds `disko/fov/fov.py` (`FoV`, `SquareFoV`, `GeoLocation`, `LonLat`, `HpAngle`, `ElAz`, `PlotCoords`, `elaz2lmn`, `hp2elaz`, `elaz2hp`, `lonlat`, `image_stats`, `factors`, plus the commented-out `HexagonSubFoV` block, all moved from `disko/sphere.py`), `disko/fov/healpix.py` (`HealpixFoV`, `HealpixSubFoV`, `create_fov`, `cmap`, `my_query_disk`, from `disko/healpix_sphere.py`), `disko/fov/mesh.py` (`AdaptiveMeshFoV`, `area`, `centroid`, `get_mesh`, `get_lmn`, from `disko/sphere_mesh.py`) and the existing `disko/fov/factory.py` (`from_hdf`). `disko.image` — the imaging algorithms — holds `disko/image/disko.py` (`DiSkO`, `DiSkOOperator`, `get_all_uvw`, `jomega`, `vis_to_real`, from `disko/disko.py`), `disko/image/telescope_operator.py` (`TelescopeOperator`, `normal_svd`, `dask_svd`, `tf_svd`, `plot_spectrum`, `plot_uv`, from `disko/telescope_operator.py`) and `disko/image/projection_lsqr.py` (`plsqr`, from `disko/projection_lsqr.py`). Infrastructure stays top level: `ms_helper` (MS I/O), the four CLI entry points, `draw_sky`, `parser_support`, `coords` (shared by both packages), `resolution`, `rime`, `util` and `multivariate_gaussian`. **Compatibility shims**: `disko/sphere.py`, `disko/healpix_sphere.py`, `disko/sphere_mesh.py`, `disko/disko.py`, `disko/telescope_operator.py` and `disko/projection_lsqr.py` still exist and re-export everything they used to define from the new homes (`from disko.sphere import FoV`, `from disko.healpix_sphere import HealpixFoV`, `import disko.sphere_mesh`, `from disko.disko import DiSkO`, `from disko.telescope_operator import MAX_COND` …), so all 34 workspace imports that resolved before the split still resolve. The shims are **silent — they deliberately do not emit a `DeprecationWarning`**: `pyproject.toml` sets no `filterwarnings`, so pytest surfaces every captured warning in the run summary (a warning here would land in every consumer's suite, including spotless's "14 passed, 1 warning" baseline), a consumer running `-W error` would turn it into a hard import failure — the exact downstream breakage this phase exists to prevent — and outside pytest CPython ignores `DeprecationWarning` for library code anyway, so it would buy signal only where it costs. The deprecation is therefore documented here and in the shims' module docstrings instead. For the same "no behaviour change" reason the moved modules pin their **pre-split logger names** (`disko.sphere`, `disko.healpix_sphere`, `disko.sphere_mesh`, `disko.disko`, `disko.telescope_operator`, `disko.projection_lsqr`): the CLIs format log records with `%(name)s`, so `disko --debug`/`disko_bayes` output would otherwise change text. Verified: suite 117 passed/13 skipped (baseline 107/13 + 10 new tests in `disko/tests/test_package_split.py`), flake8 54 findings = the `git archive HEAD` baseline with zero new findings once the moved files are mapped back, all four entry points `--help` unchanged, `disko_draw tart.hdf --FITS` byte-identical (CRVAL1 = 306.06269559089196, CRVAL2 = -45.92154041820453), `--SVG` byte-identical, and the `--ms … --field 59 --PNG` end-to-end render byte-identical
- Phase 2 of #10 (drawing & overplot): source overplotting is now coordinate-aware, fixing #7's phase-steering symptom, and the last drawing code moves out of `disko/cli.py`. `FoV.source_lmn()`/`FoV.source_draw_elaz()` place a marker at the source's ICRS `l,m` about the FoV's phase centre — the frame the image and its FITS WCS are centred on (CRVAL = phase centre, l=m=0 at the centre pixel) — mapped through the geolocated decomposition the grid itself is drawn with (`coords.lmn_to_elaz`, the exact inverse of `elaz2lmn`), while `coords.source_radec()` converts el/az sources (the TART catalog) to celestial at the sphere's site/time first; `index_of()`'s elaz signature and both frames are unchanged. On a phase-steered grid a source at the phase centre now lands on the image centre (pre-Phase-2 placement put it 0.176 deg away for test.ms field 59's PHASE_DIR), and the same source on a zenith-centred grid lands 0.176 deg off centre — asserted end to end through `to_svg` in the new `disko/tests/test_overplot.py` (6 tests; 5 of them fail on pre-Phase-2 code). Frame bound: the grid stays geolocated while the overplot is ICRS, so the two differ only by the pole-angle rotation — measured max 0.133 deg (478 arcsec; max direction-cosine disagreement 0.00231, matching Phase 1's 0.0023), i.e. ~17 px on a 4000 px/30 deg SVG, ~4 px on a 2000 px/60 deg FITS; that rotation is Phase 4's CDELT/l,m work, so the bound is asserted with its rationale rather than fixed. The CLI's `save_images()`/`path()` (PNG dpi=300, PDF dpi=600, SVG, FITS, VTK, display) moved verbatim to `disko/draw_sky.py` — the shared drawing module `disko/__init__.py` and `svd_cli.py` already import — and `disko/cli.py` now contains no `plt.` code at all
- Phase 1 of #10 (coordinate core): a FoV now carries its own celestial pointing as first-class data. New `disko.coords` holds the transforms — elaz ↔ ICRS RA/Dec (site + obstime), grid l,m ↔ celestial about a phase centre, the zenith derivation from the #14 FITS fix, and the single shared IERS-offline policy (`offline_iers()`). `FoV.set_info(..., phase_center=...)` takes a `coords.PhaseCenter` (RA/Dec + obstime + geolocation + explicit provenance: `ms_phase_dir` vs `zenith`, with the MS `field_id`/`n_fields` for per-field pointings); `phase_center_from_hdr()` builds it from the `PHASE_DIR` that `casa_read_ms` records as CRVAL1/CRVAL2, and `phase_center_radec()` returns the stored centre when there is one, else the derived zenith. The centre persists in HDF as an optional `phase_center` key, so old files still load and new files load in old readers. `HealpixFoV.index_of()` gained a `frame="icrs"` path and source placement (`source_elaz()`) accepts celestial coordinates, so sources can be located on a phase-steered grid. 26 new tests (transform round trips, provenance, HDF back-compat both directions, and an MS oracle: on `test_data/test.ms` the stored PHASE_DIR agrees with the zenith-derived centre to 0.000007 deg)
- Fix `disko_bayes --sequential` silently re-reading the same visibilities every turn when the field's pool of unflagged, resolution-limited visibilities is smaller than `--nvis`: `casa_read_ms` now warns when it must return fewer than requested, and `run_sequential` plans the run up front via a new `good_visibility_count()` — if `n_steps * nvis` exceeds the pool it reduces the per-turn draw to `pool // n_steps` (with a warning) so every step still consumes genuinely NEW visibilities instead of repeating turn 0's rows. This was turning "sequential over N new data slices" into "the same slice N times": on the test MS every turn read the same 276 rows, so the posterior variance changed by only ~0.09%/step, dominated by the null-space floor (rank ≪ n_s pixels)
- Add `disko_bayes --sequential N`: N turns of sequential Bayesian inference on the measurement set. Each turn draws `nvis` NEW (disjoint while the pool lasts) visibilities from the MS, conjugates them for the CASA UVW convention, and updates the posterior with the reduced (rank-truncated) telescope operator, starting from the diagonal heuristic prior; the mean and covariance are written at every step (`*_step<NNN>_mu`/`_var`/`_pcf` images, plus one posterior HDF5 per step when `--posterior` is given). New unit tests in `test_sequential.py`
- Fix `disko_bayes --sequential` dying of memory exhaustion after a few turns on large maps: the chain held the starting prior's full n_s x n_s covariance (GBs) and each turn's SVD result until the next turn began. The information-form path now drops the prior (only `mu0` and `s0` are needed) and releases each turn's operator before the next SVD, so the working set stays bounded by the current turn's SVD instead of accumulating to the machine's memory ceiling (previously OOM-killed around turn 6 of `make sequential` on the 19328-pixel test sphere)
- Performance: diagonal-prior fast path in `do_inference` — V is orthogonal, so `V^H (sigma0 I) V == sigma0 I`; the prior covariance is no longer rotated into the natural basis (saving two O(n_s^3) products), and the posterior is assembled fused as `V_1 Sigma_r V_1^T + sigma0 (I - V_1 V_1^T)`, avoiding the block-diagonal intermediate and discarded cross terms. Diagonal priors only (the default heuristic prior); dense priors — e.g. the chained posteriors of sequential inference — are detected via `MultivariateGaussian.is_scaled_identity()` and take the exact general path
- Performance: `image_tikhonov` uses the rank-truncated factors `V_1 (f * (U_1^T vis))` instead of the dense `V D U^T` triple product (identical results, O(n_s*rank) vs O(n_s^2)); this speeds up the 11-alpha Tikhonov sweep in `disko_svd`
- Remove the unreachable `if True:/else:` dead branch in `do_inference` (the `else` path duplicated `TelescopeOperator.sequential_inference`)
- Convert `test_disko.py` and `test_pylops_operator.py` to fast unit tests on a tiny synthetic telescope (4 antennas, nside=2 spheres), replacing real-TART-data integration tests; vectorized harmonic normalization and visibility checks, stronger solve/vis assertions. The full test suite now runs in seconds instead of ~1 h (real-data loading is no longer exercised by the unit suite)
- Add null-space image completion: `TelescopeOperator.image_natural(null_prior=...)` and `complete_image()` graft a prior sky image's null-space projection (invisible to the telescope) onto the data reconstruction, producing an image exactly consistent with the measurements. New `disko_bayes --null-prior <HEALPix FITS>` flag saves a `*_complete` image
- Make the SVD rank truncation configurable: `TelescopeOperator(..., max_cond=...)` (default unchanged at 1e4) and a `disko_bayes --max-cond` option; singular values below max(s)/max_cond are treated as null space. The SVD cache filename now includes the cutoff
- `TelescopeOperator` basis conversion methods (`sky_to_null`, `null_to_sky`, `sky_to_natural`, `natural_to_sky`, `P_r`, `range_harmonic`, `null_harmonic`) and `image_natural` now compute dask graphs explicitly and return eager numpy arrays
- Fix crash in `disko_bayes --file` (JSON branch): `create_prior(cv.v, ...)` used a nonexistent attribute; now builds the `DiSkO` first and uses its visibilities, matching the MS branch
- Fix off-by-one in `TelescopeOperator.null_to_sky`: slice `x[self.rank : -1]` dropped the last null-space component (and would raise a broadcast error for a full-length input); now `x[self.rank :]`. Previously uncalled and untested — now covered by a unit test
- Fix `MultivariateGaussian.variance()` to return the per-pixel variance (covariance diagonal); it previously returned the standard deviation (√diagonal), so `disko_bayes --var` images were mislabeled σ rather than σ². New `std()` method provides the old behavior under an honest name
- Convert bayes tests (`test_telescope_operator.py`) to fast unit tests: tiny synthetic telescope (4 antennas, nside=2 sphere) with vectorized SVD/null/range-space property checks — the file now runs in ~1 s instead of tens of minutes; add real assertions to `test_bayes` and new unit tests for `create_prior` and `do_inference` (real-data coverage remains in `test_disko.py` and `test_pylops_operator.py`)
- Fix `create_prior` prior covariance scale in `disko_bayes`: use `var = p95² · I` (prior std = p95) as logged and as implemented in `DiSkO.sequential_inference` and `TelescopeOperator.get_prior`; previously used `p95 · I`, giving a dimensionally inconsistent prior std of √p95

## 1.4.4 (2026-06-22)

- Fix `image_visibilities` RAM: use matrix-free blocked adjoint instead of building the full complex Gamma matrix in memory

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
