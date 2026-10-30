"""Unit tests for utils/ofed.py (the OFED version DOCA and MOFED record).

parse_ofed_version was compared with bash's sed/awk/tr pipeline in WSL.
run_capture and record_version are mocked, so no ofed_info runs.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest
from unittest import mock

from utils import ofed

_OFED_OUT = "OFED-internal-25.10-1.7.1:\n\nInstalled Packages:\n"
_OP = "install-doca"
_ENV = {"DISTRIBUTION": "ubuntu24.04"}


class ParseTests(unittest.TestCase):
    def test_parse_ofed_version(self):
        parse = ofed.parse_ofed_version
        self.assertEqual(parse(_OFED_OUT), "25.10-1.7.1")
        self.assertEqual(parse("OFED-internal-24.10-0.7.0:\n"), "24.10-0.7.0")
        self.assertEqual(parse("a-b-c-d-e\nsecond\n"), "c-d")
        self.assertEqual(parse("x-y-1.2:-3.4:"), "1.2-3.4")
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("only-two"))
        self.assertEqual(parse("a-b-:-:"), "-")   # as bash: only "" is rejected


class RecordTests(unittest.TestCase):
    def _record(self, captured, recorded=True):
        with mock.patch.object(ofed, "run_capture", return_value=captured) as capture, \
             mock.patch.object(ofed, "record_version", return_value=recorded) as record, \
             mock.patch.object(ofed, "log_error") as error:
            result = ofed.record_ofed_version(_ENV, _OP)
        return result, capture, record, error

    def test_records_the_version_ofed_info_reports(self):
        result, capture, record, error = self._record((0, _OFED_OUT))
        self.assertTrue(result)
        capture.assert_called_once_with(["ofed_info"], _OP, env=_ENV)
        record.assert_called_once_with("OFED", "25.10-1.7.1", _OP)
        error.assert_not_called()

    def test_failing_or_unparsable_ofed_info_is_an_error(self):
        for captured in ((127, ""), (0, "MLNX_OFED_LINUX-24.10\n")):
            with self.subTest(captured=captured):
                result, _, record, error = self._record(captured)
                self.assertFalse(result)
                record.assert_not_called()
                self.assertEqual(error.call_args.kwargs["exit_code"], captured[0])
                self.assertEqual(error.call_args.kwargs["first_line"],
                                 captured[1].split("\n", 1)[0])

    def test_failing_record_version_is_returned(self):
        self.assertFalse(self._record((0, _OFED_OUT), recorded=False)[0])


if __name__ == "__main__":
    unittest.main()
