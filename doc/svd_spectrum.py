#!/usr/bin/env python
"""
Regenerate the singular-value spectrum figure used in
doc/disko_operator_imaging.tex.

The figure shows the singular values of the real telescope operator Gamma
(disko/telescope_operator.py) for the small synthetic telescope used by the
unit tests: 4 antennas -> 12 baselines -> 24 real visibility rows, imaging a
48-pixel HEALPix sphere (nside=2). The vertical red line marks the rank
truncation: singular values at or to the right of the line are treated as
null space of the telescope (below max(s)/max_cond with max_cond=1e4).

Usage:
    python doc/svd_spectrum.py            # writes doc/svd_spectrum.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disko import DiSkO, HealpixFoV, TelescopeOperator

OUT = Path(__file__).resolve().parent / "svd_spectrum.png"


def main():
    np.random.seed(42)

    # A tiny synthetic telescope: 4 antennas -> 12 baselines -> 24 real
    # visibility rows. The sphere has 48 pixels, so n_s > n_v and Gamma has
    # an exact null space (see disko/tests/test_telescope_operator.py).
    frequency = 1.5e9
    ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
    disko = DiSkO.from_ant_pos(ant_pos, frequency=frequency)
    sphere = HealpixFoV(nside=2)

    to = TelescopeOperator(disko, sphere)
    print(f"n_v={to.n_v}, n_s={to.n_s}, rank={to.rank}, null={to.n_n()}")

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=200)
    ax.plot(to.s, "o-", ms=3, lw=1)
    ax.axvline(to.rank - 0.5, color="red", ls="--", lw=1,
               label=f"rank truncation (r={to.rank})")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Index $i$")
    ax.set_ylabel("Singular value $\\sigma_i$")
    ax.set_title(f"Singular value spectrum $N_v$={to.n_v}, "
                 f"$N_s$={to.n_s}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
