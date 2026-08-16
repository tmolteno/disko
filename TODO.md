# TODO — disko-bayes remaining issues

From the code review of the disko-bayes logic. Issues 1 (prior scale),
3 (`variance()` returning std), and 7 (`null_to_sky` off-by-one) are
fixed; the rest are open.

## 1. `--show-sources` crashes with NameError

`handle_output` (`disko/bayes_cli.py:219`) references `src_list`, but it
is only assigned when the flag is *off* (`if not ARGS.show_sources:
src_list = None`). When the flag is *on*, the value computed in
`handle_bayes` is never passed through, so plotting raises `NameError`.

**Fix:** pass `src_list` as a parameter to `handle_output` instead of
relying on an undefined global. Easy.

## 2. JSON branch images only the last snapshot

The loop at `disko/bayes_cli.py:136-141` overwrites `cv` on every
iteration of `calib_info["data"]` and runs inference once, on the final
snapshot. Unlike the HDF5 branch (`bayes_cli.py:167-169`) there is no
sequential chaining (posterior -> prior) across snapshots.

**Fix:** either chain inference across snapshots like the HDF5 branch,
or run and save each snapshot independently. Decide the intended
behaviour first. Easy once decided.

## 3. Dead code in the bayes path

- `do_inference` (`disko/bayes_cli.py:75-106`) has an `if True:` ...
  `else:` where the `else` branch (`to.sequential_inference`) is
  unreachable and duplicates the live branch. Collapse it.
- `_original_positions = deepcopy` (`bayes_cli.py:128`) assigns the
  function itself instead of calling it (hidden by a `noqa`). Remove.
- `TelescopeOperator.n_s()` / `n_v()` methods
  (`disko/telescope_operator.py:296-300`) are shadowed per-instance by
  the integer attributes of the same name set in `__init__`, so they
  can never be called. Remove the methods or rename the attributes.

Trivial cleanup.

## 4. Real/imag noise covariance is a heuristic

`do_inference` (`disko/bayes_cli.py:69`) builds the visibility noise
covariance as `[[diag, 0.5*diag], [0.5*diag, diag]]` with a TODO
acknowledging that a proper covariance linking the real and imaginary
components is not implemented. The 0.5 coupling is a guess, not derived.

**Fix:** derive the correct cross-covariance for complex Gaussian
visibility noise (Re/Im are independent and equal-variance for circular
noise, giving zero off-diagonal blocks) or make the coupling
configurable. Needs a statistics decision; verify against simulated
noise. Medium.

## 5. Later: same fast-unit-test treatment for the remaining slow files

`test_disko.py` and `test_pylops_operator.py` still run against real
TART data with `HealpixFoV(16)` (3072 pixels) and dominate the ~1 h full
suite (CI runs plain `pytest` on every push). The pattern from
`test_telescope_operator.py` (tiny synthetic telescope + vectorized
property checks) applies directly. Medium effort.
