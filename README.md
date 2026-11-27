# Discrete Sky Operator (DiSkO) Synthesis Imaging

[![Build Status](https://travis-ci.org/tmolteno/disko.svg?branch=master)](https://travis-ci.org/tmolteno/disko)

Author: Tim Molteno tim@elec.ac.nz

Its so cool its POINTLESS. Image by using the telescope operator keeping track of the telescope null-space and range-space. DiSkO uses a discrete representation of the field of view (as a healpix grid, or unstructured mesh) in the sky space. This means we can image arbitratily wide fields of view (including the full sphere), as well as arbitrarily shaped fields (circles are preferred).

DiSkO can perform sparsity reduction by regularization and controls the volume of the sky solution. The result is an imaging algorithm that is sensitive to diffuse broad sources, and does not require restoration like CLEAN. Publications to appear :)

## Howto

`disko` picks sensible defaults (issue #12): with no `--mesh`/`--healpix` it
images on a healpix grid, and with no algorithm flag it uses Tikhonov
regularization with `--alpha 0.02`. So the shortest command that images a
measurement set is

    disko --ms test_data/test.ms --nside 32 --HDF fov.h5

Write the field of view to HDF, then draw it with `disko_draw` (which
also does the source overplotting, `--show-sources`):

    disko_draw fov.h5 --show-sources --SVG draw.svg --PNG draw.png

Spelled out in full, that first command's defaults are

    disko --ms test_data/test.ms --healpix --tikhonov --alpha 0.02 --nside 32 --HDF fov.h5

To load a data from a measurement set 

    disko --ms test_data/test.ms --nside 32 --PDF

Other solvers: `--lasso` (L1), `--matrix-free` with one of `--lsqr`, `--lsmr`
or `--fista`, or `--no-regularization` to solve the visibilities directly
(that unregularized solve is what you used to get by default, which is why
the images were poor).

### Beam patterns

By default every antenna is taken to respond equally over the whole sky.
`--beam` gives the telescope operator a primary beam instead, so that the
image is of the sky before the beam:

    disko --ms test_data/test.ms --nside 32 --beam tart --HDF fov.h5

Nothing else changes: draw the field of view with `disko_draw` as above. The
beam is part of the operator, so every solver (`--tikhonov`, `--lasso`,
`--matrix-free`, `--no-regularization`) images through it. `disko_bayes`
takes the same option, on every turn of `--sequential` too:

    disko_bayes --ms test_data/test.ms --nside 32 --beam tart --mu --var --PNG

`--beam` takes the same values as `spotless` and `tartball`, which read them
from [tart-beam](https://github.com/tart-telescope/beams):

| Value | Meaning |
|-------|---------|
| `tart` (or `default`) | The built-in TART beam: roughly uniform above 10° elevation, falling to zero at the horizon |
| A JSON file with `coeffs` | A beam saved by `tart_beam.Beam.to_json()` |
| A JSON list of `el`, `az`, `gain` | Samples in degrees, fitted with a spherical-harmonic pattern when they are loaded |

A bare `--beam` means `tart`.

A beam of your own is a list of the gain at measured or simulated directions,
with enough of them to cover the pattern (a grid, as in
[tart-beam's example](https://github.com/tart-telescope/beams/blob/main/examples/sample_beam.json)):

    [
      {"el": 90.0, "az": 0.0,  "gain": 1.0},
      {"el": 45.0, "az": 30.0, "gain": 0.42},
      ...
    ]

    disko --ms test_data/test.ms --nside 32 --beam my_beam.json --HDF fov.h5

Such a list is fitted every time it is loaded. To fit it once, save the
result and give that instead:

    from tart_beam import fit_from_json
    fit_from_json("my_beam.json", degree=8, q=2).to_json("fitted_beam.json")

A list is taken to be zenith-pointed: a sample at elevation `el` sits at
`w = sin(el)`, and samples below the horizon are dropped. Behind the antenna
the gain is zero.

What you get:

- The field of view written with `--HDF` records the beam, one gain per pixel,
  in a `beam` dataset.
- With `--PNG`, `--PDF`, `--SVG` or `--FITS`, `disko` also writes a
  `*_beam_*` image of the pattern over the field of view.
- Pixels the beam gives no gain are exactly null, so nothing is imaged
  behind the antenna. On a full sphere a planar array cannot otherwise tell
  the sky from its reflection below the ground.

From Python, `FoV.set_beam()` puts the beam on a field of view, and every
operator built from it (`DiSkO.make_gamma()`, `DiSkOOperator`,
`TelescopeOperator`) carries it:

    from disko import HealpixFoV, TelescopeOperator, load_beam

    sphere = HealpixFoV(32)
    sphere.set_beam(load_beam("tart"))     # or an array of one gain per pixel
    to = TelescopeOperator(tel, sphere)    # the operator scaled by the beam

where `tel` is the `DiSkO` for the observation.

Things to know:

- The gain is clipped at zero and has no floor, and nothing is divided by
  it. Where the beam varies within a resolution element (the lowest 10° of a
  TART field) the de-beamed fluxes depend on the regularizer; read them with
  that in mind.
- `--no-regularization` standardizes its image, so there the pixels outside
  the beam hold a constant instead of zero. In `disko_bayes` they hold the
  prior mean (`--mu`) and the prior variance (`--var`).
- On the adaptive mesh (`--mesh --adaptive`) the beam is sampled again after
  each refinement.
- `disko_svd --beam` is an error. Its old `--beam` flag is now `--psf`, which
  images the synthesised beam.

## VLA imaging with DiSkO

Download the VLA 5GHZ continuum survey measurement set. AG733_A061209.xp1 from the NRAO site.

Calibrate and then split the measurement set, following the CASA tutorial [https://casaguides.nrao.edu/index.php/VLA_5_GHz_continuum_survey_of_Seyfert_galaxies]

    disko --fov 0.01deg --ms NGC1194.split.ms --SVG --res 0.15arcsec --tikhonov
## More challenging

This tutorial should generate a file with lots of diffuse radiation. 
[https://casaguides.nrao.edu/index.php/VLA_Continuum_Tutorial_3C391-CASA5.5.0]

    wget http://casa.nrao.edu/Data/EVLA/3C391/3c391_ctm_mosaic_10s_spw0.ms.tgz
    wget https://github.com/jaredcrossley/CASA-Guides-Script-Extractor/blob/master/extractCASAscript.py
    
    python extractCASAscript.py 'https://casaguides.nrao.edu/index.php?title=VLA_Continuum_Tutorial_3C391-CASA5.5.0'

Then in CASA

    execfile("VLAContinuumTutorial3C391-CASA5.5.0.py")
    
This should generate a suitable measurement set to image.

## Releasing

Releases are cut with `uv run invoke release`: exhaustive tests on every
supported Python, a version bump, publication to PyPI through CI, and then the
downstream tools image and tart_cargo releases in the same chain. See
[RELEASE.md](RELEASE.md) for the process, and the matching documents in
[tools](https://github.com/tart-telescope/tools/blob/main/RELEASE.md) and
[tart_cargo](https://github.com/tart-telescope/tart_cargo/blob/main/RELEASE.md).

## TODO

* Add a --full-sphere option which fixes the sphere in celestial coordinates, and then points the phase center of an observation correctly. Requires a beam pattern to be specified (or at least a hemispherical beam) -- `--beam` now provides one. A beam is a sky vector mask. I.e., should fall to zero 'outside' the beam.
* Deal with flagging, output residuals in a way that can be used to flag in the measurement set. This means some casa expression that changes the MS.
