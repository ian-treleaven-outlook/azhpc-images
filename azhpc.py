#!/usr/bin/env python3
"""Python implementation of the Azure HPC CLI tool."""

from datetime import datetime, timezone

VERSION = "0.1.0"
BUILD_COMMIT = "dev"
BUILD_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")