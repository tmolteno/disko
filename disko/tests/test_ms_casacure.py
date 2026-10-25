#
# Issue #23: disko reads measurement sets through casacure (the Rust
# replacement for casacore) rather than python-casacore, so that it no
# longer depends on dask-ms installing python-casacore for it (dask-ms 0.3
# drops python-casacore in favour of arcae).
#
# The only API that differs between the two is table.query(): python-casacore
# takes (where, sortlist=, columns=) and builds a TaQL command from them,
# casacure takes only a selection expression. _taql_select() has to build
# exactly the command python-casacore would have, or rows/columns/order could
# silently differ -- hence the explicit contract test below.
#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License GPLv3
#

import os
import unittest

from disko import ms_helper
from disko.ms_helper import _taql_select, good_visibility_count
from disko.resolution import Resolution

MS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "test.ms")


class _RecordingTable:
    """Stands in for a casacure table: records the TaQL it is asked to run."""

    def __init__(self):
        self.commands = []

    def taql(self, cmd):
        self.commands.append(cmd)
        return "RESULT"


class TestTaqlSelect(unittest.TestCase):
    def test_reproduces_the_python_casacore_query_command(self):
        # python-casacore: ms.query("FIELD_ID==0", sortlist="ARRAY_ID",
        #                           columns="TIME, DATA")
        # builds: select TIME, DATA from $1 where FIELD_ID==0 orderby ARRAY_ID
        t = _RecordingTable()
        out = _taql_select(
            t, where="FIELD_ID==0", columns="TIME, DATA", sortby="ARRAY_ID"
        )
        self.assertEqual(out, "RESULT")
        self.assertEqual(
            t.commands,
            ["select TIME, DATA from $1 where FIELD_ID==0 orderby ARRAY_ID"],
        )

    def test_omits_orderby_when_no_sort_was_asked_for(self):
        t = _RecordingTable()
        _taql_select(t, where="FIELD_ID==1", columns="UVW")
        self.assertEqual(t.commands, ["select UVW from $1 where FIELD_ID==1"])

    def test_column_list_is_passed_through_verbatim(self):
        # casa_read_ms projects exactly these columns (and the weight query
        # asks for WEIGHT_SPECTRUM alone).
        t = _RecordingTable()
        _taql_select(
            t,
            where="FIELD_ID==3",
            columns="TIME, DATA, UVW, ANTENNA1, ANTENNA2, FLAG",
            sortby="ARRAY_ID",
        )
        self.assertIn("TIME, DATA, UVW, ANTENNA1, ANTENNA2, FLAG", t.commands[0])


class TestCasacureIsTheTableBackend(unittest.TestCase):
    def test_ms_helper_imports_table_from_casacure(self):
        # Guards against python-casacore coming back in unnoticed: dask-ms
        # 0.2.x still installs it, so a stray "from casacore.tables import
        # table" would keep working in tests and only break at dask-ms 0.3.
        module = getattr(ms_helper.table, "__module__", "")
        self.assertTrue(
            module.startswith("casacure"),
            f"ms_helper.table comes from {module!r}, expected casacure",
        )

    @unittest.skipUnless(os.path.isdir(MS_PATH), "test_data/test.ms fixture missing")
    def test_good_visibility_count_reads_through_casacure(self):
        # 276 unflagged, resolution-limited rows in field 0 of the fixture at
        # 1 deg resolution -- the same number python-casacore returned, so the
        # TaQL translation selects the same rows.
        n = good_visibility_count(
            MS_PATH, angular_resolution=Resolution.from_string("1deg").degrees()
        )
        self.assertEqual(n, 276)


if __name__ == "__main__":
    unittest.main()
