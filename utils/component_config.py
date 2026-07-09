"""component_config.py — resolve a component's config from versions.json.

Python port of get_component_config in utils/utilities.sh. It walks the same
lookup hierarchy, most specific first:

  1. component.distribution.architecture.<gpu_sku>.<node_type>
  2. component.distribution.architecture.<gpu_sku>.default
  3. component.distribution.architecture.<gpu_sku>   (legacy direct config)
  4. component.distribution.architecture
  5. component.common

Unlike the Bash version (which echoes a JSON string that callers re-parse with
jq), this returns the resolved config as a dict — or None when nothing matches.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Markers that indicate a <gpu_sku> node is a *nested* config (keyed by node
# type) rather than a direct config. Mirrors the Bash has_nested_config check.
_NESTED_MARKERS = ("default", "baremetal", "azure_vm")

# Where installed component versions are recorded (mirrors the Bash
# write_component_version in utils/utilities.sh).
_VERSIONS_FILE = "/opt/azurehpc/component_versions.txt"


def normalize_key(value: str) -> str:
    """Mirror normalize_component_config_key: lowercase and collapse any run of
    non-alphanumeric characters into a single underscore."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower())


def get_component_config(component, versions, *, distribution, architecture,
                         gpu=None, sku=None, node_type="azure-vm"):
    """Resolve the config for `component` from the `versions` dict.

    Returns the matching config dict, or None if the hierarchy yields nothing.
    """
    comp = versions.get(component)
    if not isinstance(comp, dict):
        return None

    config = None

    if gpu and sku:
        sku_key = normalize_key(f"{gpu}_{sku}")
        node_type_key = normalize_key(node_type or "azure-vm")

        dist_node = comp.get(distribution)
        arch_node = dist_node.get(architecture) if isinstance(dist_node, dict) else None
        sku_config = arch_node.get(sku_key) if isinstance(arch_node, dict) else None

        if isinstance(sku_config, dict):
            # 1. <gpu_sku>.<node_type>
            if sku_config.get(node_type_key) is not None:
                config = sku_config[node_type_key]
            # 2. <gpu_sku>.default
            elif sku_config.get("default") is not None:
                config = sku_config["default"]
            # 3. <gpu_sku> as a direct config (only if not a nested node)
            elif not any(marker in sku_config for marker in _NESTED_MARKERS):
                config = sku_config

    # 4. architecture level
    if config is None:
        dist_node = comp.get(distribution)
        arch_config = dist_node.get(architecture) if isinstance(dist_node, dict) else None
        if arch_config is not None:
            config = arch_config

    # 5. common
    if config is None:
        config = comp.get("common")

    return config


def write_component_version(component, version, path=_VERSIONS_FILE):
    """Record an installed component's version.

    Python port of write_component_version in utils/utilities.sh: maintains a
    JSON file mapping component -> version (default
    /opt/azurehpc/component_versions.txt), creating it or updating it in place.
    Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    if not isinstance(data, dict):
        data = {}

    data[component] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)
    return path
