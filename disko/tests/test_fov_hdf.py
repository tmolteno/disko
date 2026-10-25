#
# Issue #12: the README's flagship command
#     disko --ms test_data/test.ms --nside 32 --HDF fov.h5
# used to die after imaging with
#     TypeError: Object dtype dtype('O') has no native HDF5 equivalent
# because HealpixSubFoV built from --nside leaves res_arcmin=None, and
# to_hdf() wrote that None as an object-dataset. Pin the round trip.
#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License GPLv3
#

import os
import tempfile
import unittest

from disko.fov.factory import from_hdf
from disko.fov.healpix import create_fov
from disko.resolution import Resolution


def make_fov_from_nside(nside=16):
    """The way create_fov builds a FoV when --nside is given (res_arcmin is
    never passed, which is exactly what left it None)."""
    return create_fov(
        nside,
        fov=Resolution.from_string("10deg"),
        res=Resolution.from_string("1deg"),
        theta=0.0,
        phi=0.0,
    )


class TestHealpixHdfRoundTrip(unittest.TestCase):
    def test_res_arcmin_is_recorded_when_built_from_nside(self):
        fov = make_fov_from_nside()
        self.assertIsNotNone(fov.res_arcmin)
        self.assertGreater(fov.res_arcmin, 0.0)

    def test_to_hdf_from_nside_succeeds(self):
        fov = make_fov_from_nside()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fov.h5")
            # Used to raise TypeError (object dtype has no HDF5 equivalent).
            fov.to_hdf(path)
            self.assertTrue(os.path.exists(path))

    def test_round_trip_preserves_pixels_and_resolution(self):
        fov = make_fov_from_nside()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fov.h5")
            fov.to_hdf(path)

            back = from_hdf(path)
            self.assertEqual(back.nside, fov.nside)
            self.assertEqual(back.npix, fov.npix)
            self.assertAlmostEqual(back.res_arcmin, fov.res_arcmin)
            self.assertEqual(len(back.pixels), len(fov.pixels))


if __name__ == "__main__":
    unittest.main()
