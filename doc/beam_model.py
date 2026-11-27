#!/usr/bin/env python
"""
Regenerate the beam-model figure used in doc/disko_operator_imaging.tex
(doc/beam_model.png).

The figure shows what a beam does to a planar array on a full-sphere FoV.
Without one, a direction and its mirror image below the horizon have
identical harmonics, and the Tikhonov image splits a source with its mirror.
With the base TART beam the mirror pixels are exactly null.

The beam is set on a copy of the sphere with FoV.set_beam(): disko then
weights each harmonic by b_j A_j, which is exactly Gamma diag(b).

The beam is the profile that tart_beam.base_tart_beam() is fitted to: uniform
above 10 degrees elevation, a cosine half-cycle down to zero at the horizon
in w = sin(el), zero behind the antenna. It is written out here so the script
is exact and needs no fitted pattern. The built-in pattern itself,
    sphere.set_beam(disko.load_beam("tart"))
is a least-squares fit to it and differs by about 2 per cent.

Usage:
    python doc/beam_model.py            # writes doc/beam_model.png
"""

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np

from disko import DiSkO, HealpixFoV, TelescopeOperator

OUT = Path(__file__).resolve().parent / "beam_model.png"

FREQUENCY = 1.57542e9
"""TART's operating frequency [Hz]."""
ANTENNAS = np.array([
    [-0.195684,  0.832998], [ 0.124652,  0.816187],
    [-0.321227,  0.694401], [ 0.528114,  1.268005],
    [-0.186441,  1.409113], [ 0.273175,  1.383467],
    [ 1.032164, -0.433371], [ 1.190631, -0.153561],
    [ 1.102020, -0.613829], [ 0.982554,  0.409603],
    [ 0.526444, -0.149723], [ 0.768389,  0.243472],
    [-0.426891, -1.025999], [-0.155886, -1.182349],
    [-0.604371, -1.090551], [ 0.411615, -0.965206],
    [-0.167071, -0.524431], [ 0.232578, -0.748506],
    [-0.856204,  0.373012], [-1.120715,  0.217621],
    [-0.821241,  0.558143], [-1.202041, -0.369674],
    [-0.552788, -0.108693], [-0.945059, -0.326391],
])
"""East and north offsets [m] of the 24 antennas of test_data/test_data.json.

A planar array: the longest baseline is 2.6 m, which resolves 5.1 deg
(Rayleigh) at FREQUENCY.
"""
TRANSITION_DEG = 10.0
"""Elevation below which the base TART beam tapers to zero [deg]."""
SOURCE_ELEVATION_DEG = 20.0
SOURCE_AZIMUTH = np.radians(40.0)
ALPHA = 0.003
"""Tikhonov strength, TelescopeOperator.image_tikhonov()."""
NSIDE = 16


def disko():
    """The TART array, as a DiSkO telescope."""
    ant = np.hstack((ANTENNAS, np.zeros((len(ANTENNAS), 1))))
    return DiSkO.from_ant_pos(ant, frequency=FREQUENCY)


def base_tart_beam(el_r, transition_deg=TRANSITION_DEG):
    """The base TART beam at elevations el_r [rad], pointed at the zenith.

    1 above the transition, 0.5 * (1 - cos(pi * w / w_t)) between the horizon
    and the transition with w = sin(el), and 0 below the horizon.
    """
    el_deg = np.degrees(el_r)
    gain = np.ones_like(el_r)
    taper = el_deg < transition_deg
    w_frac = np.clip(np.sin(el_r[taper]) / np.sin(np.radians(transition_deg)), 0.0, 1.0)
    gain[taper] = 0.5 * (1.0 - np.cos(np.pi * w_frac))
    return gain


def with_beam(sphere, beam):
    """A copy of the sphere carrying a beam: one gain per pixel."""
    ret = sphere.copy()
    ret.set_beam(beam)
    return ret


def side_view(ax, sphere, values, title, norm, cmap, mark=None):
    """The half of the sphere nearest SOURCE_AZIMUTH, seen from the side.

    Every pixel is drawn at (sin(az - az0) cos(el), sin(el)): the zenith is at
    the top, the horizon across the middle, and a direction and its mirror
    image below the ground are reflections of each other in the horizon.
    """
    front = np.cos(sphere.az_r - SOURCE_AZIMUTH) >= 0
    x = np.sin(sphere.az_r - SOURCE_AZIMUTH) * np.cos(sphere.el_r)
    y = np.sin(sphere.el_r)
    ax.axhspan(-1, 0, color="0.92", zorder=0)
    ax.scatter(x[front], y[front], c=np.asarray(values)[front], s=11, marker="s",
               cmap=cmap, norm=norm, linewidths=0, zorder=2)
    ax.axhline(0, color="0.25", lw=0.8, zorder=3)
    if mark is not None:
        # Wide enough to leave the marked pixel visible inside it.
        ax.plot(x[mark], y[mark], "o", mfc="none", mec="k", ms=13, mew=0.8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], ["-90", "-45", "0  (horizon)", "45", "90"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)
    ax.text(0, 1.1, "zenith", ha="center", fontsize=8, color="0.25")
    ax.text(0, -1.19, "behind the antenna", ha="center", va="top", fontsize=8, color="0.25")
    ax.set_ylabel("Elevation [deg]", fontsize=9)


def main():
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")

    sphere = HealpixFoV(NSIDE)
    beam = base_tart_beam(sphere.el_r)
    j = sphere.index_of(np.radians(SOURCE_ELEVATION_DEG), SOURCE_AZIMUTH)
    mirror = sphere.index_of(np.radians(-SOURCE_ELEVATION_DEG), SOURCE_AZIMUTH)
    sky = np.zeros(sphere.npix)
    sky[j] = 1.0 / sphere.pixel_areas[j]  # one unit of flux in pixel j
    above = sphere.el_r > 0

    plain, beamed = TelescopeOperator(disko(), sphere), TelescopeOperator(disko(), with_beam(sphere, beam))
    gamma, gamma_b = np.asarray(plain.gamma), np.asarray(beamed.gamma)
    assert np.allclose(gamma_b, gamma * beam[np.newaxis, :]), "A_j -> b_j A_j is not Gamma diag(b)"
    assert np.allclose(gamma[:, j], gamma[:, mirror]), "the planar array's mirror columns differ"
    assert not np.any(gamma_b[:, ~above]), "a pixel behind the horizon is not null"

    vis = gamma_b @ sky  # what the telescope with the beam measures
    image = plain.image_tikhonov(vis, sphere, ALPHA, scale=False).ravel() * sphere.pixel_areas
    image_b = beamed.image_tikhonov(vis, beamed.sphere, ALPHA, scale=False).ravel() * sphere.pixel_areas
    for label, pixels in (("no beam", image), ("beam folded in", image_b)):
        print(f"{label}: source pixel {pixels[j]:.3f}, mirror pixel {pixels[mirror]:.3f}, "
              f"flux above the horizon {np.sum(pixels[above]):.3f}, behind it "
              f"{np.sum(pixels[~above]):.3f}")
    print(f"beam: {np.count_nonzero(beam == 0)} of {sphere.npix} pixels are exactly zero")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), dpi=200)
    # Symmetric in the pixel flux and logarithmic beyond 0.02, so that the
    # source (0.17 and 0.36) and its surroundings (within 0.03) both show.
    limit = max(np.max(np.abs(image)), np.max(np.abs(image_b)))
    image_norm = SymLogNorm(linthresh=0.02, linscale=0.8, vmin=-limit, vmax=limit)
    marked = np.zeros(sphere.npix, dtype=bool)
    marked[j] = marked[mirror] = True
    side_view(axes[0], sphere, image, "No beam: the source and its mirror", image_norm,
              "RdYlBu_r", marked)
    side_view(axes[1], sphere, beam, "The beam as a sky vector", Normalize(0.0, 1.0), "viridis")
    side_view(axes[2], sphere, image_b, "Beam folded in: the mirror is null", image_norm,
              "RdYlBu_r", marked)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
