"""version_validate.py — check versions.json for internal consistency.

Step 2 of version-compatibility validation. Where version_report just *lists*
versions, this looks for combinations that cannot be right.

First rule — CUDA-major agreement: several components (HPC-X, ...) ship builds
tied to a specific CUDA major, encoded in their download URL as 'cudaNN'. That
major must match the CUDA driver the image installs. A mismatch means the image
would pull, e.g., a CUDA 12 HPC-X onto a CUDA 13 image.
"""

from __future__ import annotations

import re

from utils.component_config import get_component_config

_CUDA_IN_URL = re.compile(r"cuda(\d+)")


def active_hpcx_component(gpu, sku):
    """Which HPC-X variant the build actually installs for this target.

    Mirrors the selection in components/install_mpis.sh: AMD uses hpcx_amd,
    non-InfiniBand SKUs (NCv6) use the inbox build, everything else uses the
    default hpcx. The three are mutually exclusive, so only the selected one
    should be checked for CUDA consistency.
    """
    if gpu == "AMD":
        return "hpcx_amd"
    if sku == "NCv6":
        return "hpcx_inbox"
    return "hpcx"


def _major(version):
    """Integer major of a version string: '13.0' -> 13, else None."""
    if not isinstance(version, str):
        return None
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def cuda_major_from_config(cuda_config):
    """The CUDA major the image installs, from cuda.driver.version."""
    if not isinstance(cuda_config, dict):
        return None
    driver = cuda_config.get("driver")
    if not isinstance(driver, dict):
        return None
    return _major(driver.get("version"))


def cuda_major_in_url(url):
    """The CUDA major baked into a download URL ('...-cuda13-...') or None."""
    if not isinstance(url, str):
        return None
    match = _CUDA_IN_URL.search(url)
    return int(match.group(1)) if match else None


def check_cuda_consistency(versions, *, distribution, architecture,
                           gpu=None, sku=None, node_type="azure-vm"):
    """Return a list of violation messages where the HPC-X build the image
    installs disagrees with the resolved CUDA driver major. Empty = consistent.

    Only the HPC-X variant actually selected for this target is checked (see
    active_hpcx_component), so sibling variants that aren't installed don't
    produce false positives. Non-NVIDIA targets install no CUDA, so they return
    [] — there is nothing to check.
    """
    if gpu != "NVIDIA":
        return []

    selectors = dict(distribution=distribution, architecture=architecture,
                     gpu=gpu, sku=sku, node_type=node_type)
    cuda_major = cuda_major_from_config(get_component_config("cuda", versions, **selectors))
    if cuda_major is None:
        return []

    violations = []
    hpcx_name = active_hpcx_component(gpu, sku)
    config = get_component_config(hpcx_name, versions, **selectors)
    if isinstance(config, dict):
        url_major = cuda_major_in_url(config.get("url"))
        if url_major is not None and url_major != cuda_major:
            violations.append(
                f"{hpcx_name}: built for CUDA {url_major} but image installs "
                f"CUDA {cuda_major}"
            )
    return violations
