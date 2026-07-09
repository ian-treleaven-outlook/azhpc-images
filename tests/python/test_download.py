"""Unit tests for utils.download (native download + checksum verification).

These are fully offline: the "download" tests serve a local file over a file://
URL, so there is no network dependency and the results are deterministic.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from utils.download import (
    ChecksumError,
    download_and_verify,
    sha256_of,
    verify_checksum,
)


class ChecksumTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.file = self.dir / "data.bin"
        self.payload = b"hello azhpc"
        self.file.write_bytes(self.payload)
        self.sha = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        self._tmp.cleanup()

    def test_sha256_of_matches_hashlib(self):
        self.assertEqual(sha256_of(self.file), self.sha)

    def test_verify_checksum_match(self):
        self.assertTrue(verify_checksum(self.file, self.sha))

    def test_verify_checksum_mismatch(self):
        self.assertFalse(verify_checksum(self.file, "0" * 64))

    def test_verify_checksum_is_case_insensitive(self):
        self.assertTrue(verify_checksum(self.file, self.sha.upper()))


class DownloadAndVerifyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # A "remote" source file, served to the downloader via a file:// URL.
        self.src = self.dir / "source.tgz"
        self.payload = b"payload-1234"
        self.src.write_bytes(self.payload)
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.url = self.src.resolve().as_uri()   # file:///...
        self.dest_dir = self.dir / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def test_downloads_and_verifies(self):
        path = download_and_verify(self.url, self.sha, dest_dir=self.dest_dir)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "source.tgz")
        self.assertEqual(sha256_of(path), self.sha)

    def test_raises_on_bad_checksum(self):
        with self.assertRaises(ChecksumError):
            download_and_verify(self.url, "0" * 64, dest_dir=self.dest_dir)


if __name__ == "__main__":
    unittest.main()
