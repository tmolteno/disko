#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License GPLv3
#
"""
Antenna beam patterns for the telescope operator.

A beam pattern is the gain of the antennas in each direction. Sampled on the
pixels of a field of view it is a sky vector, one gain per pixel, and the
telescope operator becomes that of the beam-scaled telescope: the harmonics
weight pixel j by b_j A_j, so v = Gamma diag(b) s. Without one the operator
assumes every antenna responds equally over the whole sky.

The patterns come from tart_beam (github.com/tart-telescope/beams). --beam
takes the same values as spotless's and tartball's: "tart" for the built-in
TART beam, or the path of a JSON file holding either a beam saved by
tart_beam.Beam.to_json(), or a list of {el, az, gain} samples in degrees,
which is fitted when it is loaded.

disko does not divide anything by the gain, as a source fit does: the
pattern weights the operator, and the sky before the beam is solved for. So
the gain has no floor. A fitted pattern's small overshoot below zero just
above the horizon is clipped at zero instead (FoV.set_beam).
"""

import json
import logging

import numpy as np
from tart_beam import Beam, base_tart_beam, fit_from_json

logger = logging.getLogger(__name__)

BUILTIN = ("tart", "default")
"""What --beam takes for the built-in TART beam."""

BEAM_HELP = (
    "Antenna beam pattern in the telescope operator: 'tart' for the built-in "
    "TART beam, or a JSON file holding a tart_beam beam or {el, az, gain} "
    "samples to fit. The image is then of the sky before the beam."
)


def load_beam(name):
    """Load the beam pattern that --beam names.

    Args:
        name: "tart" or "default" for the built-in TART beam, or the path of a
            JSON file: a beam saved by tart_beam.Beam.to_json(), or a list of
            {el, az, gain} samples in degrees, which is fitted.

    Returns:
        A tart_beam.Beam.

    Raises:
        ValueError: For anything else, naming it.
    """
    if name in BUILTIN:
        logger.info("Using the built-in TART beam")
        return base_tart_beam(degree=8, q=1, name="TART_base")

    logger.info(f"Loading beam from {name}")
    try:
        with open(name) as f:
            payload = json.load(f)
    except OSError as e:
        raise ValueError(
            f"{name} is neither '{BUILTIN[0]}' nor a readable file ({e.strerror})"
        ) from e
    except ValueError as e:  # json.JSONDecodeError
        raise ValueError(f"{name} is not a beam file: {e}") from e
    try:
        if isinstance(payload, dict) and "coeffs" in payload:
            beam = Beam.from_json(name)
        else:
            beam = fit_from_json(name, degree=8, q=2, ridge=0.0, name="user_beam")
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as e:
        raise ValueError(f"{name} is not a beam file: {e}") from e
    return beam


def add_beam_argument(parser):
    """Declare --beam on a parser, taking the values load_beam() reads.

    A bare --beam means "tart"; without the option there is no beam. Declare
    it on each command line's own parser, not on the shared
    sphere_args_parser(): spotless uses that one as a parent parser and
    declares its own --beam.
    """
    parser.add_argument(
        "--beam",
        nargs="?",
        const=BUILTIN[0],
        default=None,
        metavar="BEAM",
        help=BEAM_HELP,
    )


def beam_from_args(args, sphere, on_error=None):
    """Load the beam that a command line's --beam names, and set it on a sphere.

    Called once the sphere exists and before any data is read: a beam that
    cannot be loaded is a mistake on the command line, and need not wait for
    the telescope.

    Args:
        args: The parsed command line.
        sphere: The field of view the command line will image on.
        on_error: A parser's error() method. Given, a beam that cannot be
            loaded is reported there, which exits with status 2.

    Returns:
        The tart_beam.Beam, or None when --beam was not given.

    Raises:
        ValueError: The beam cannot be loaded, and on_error was not given.
    """
    name = getattr(args, "beam", None)
    if name is None:
        return None
    try:
        beam = load_beam(name)
    except ValueError as e:
        if on_error is None:
            raise
        on_error(f"--beam: {e}")
        raise AssertionError("parser.error() returned") from e
    logger.info(f"Beam: {beam}")
    sphere.set_beam(beam)
    return beam
