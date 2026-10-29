"""component_config.py — resolve a component's config from versions.json.

Python port of get_component_config in utils/utilities.sh. It walks the same
lookup order as bash, most specific first:

  1. component.distribution.architecture.<node_type>.<gpu_sku>
  2. component.distribution.architecture.<node_type>.default
  3. component.distribution.architecture.<node_type>   (only if it is a config
                                                        itself, not a table of
                                                        per-SKU overrides)
  4. component.distribution.architecture
  5. component.common

Steps 1-3 only run when both GPU and SKU are known, as in bash.

  * <node_type> is the deployment target, the same values bash reads from
    TARGET_NODE_TYPE (see NODE_TYPES below; default "azure_vm_regular").
  * <gpu_sku> is the vendor and SKU joined and normalised, e.g. "NVIDIA" +
    "V100" -> "nvidia_v100".

A real example from versions.json: CUDA on Ubuntu 24.04 x86_64 is 13.0, but a
V100 on a regular Azure VM gets 12.9 because of this override:

    "cuda": { "ubuntu24.04": { "x86_64": {
        "driver": { "version": "13.0", ... },          <- step 4 for most SKUs
        "azure_vm_regular": {
            "nvidia_v100": { "driver": { "version": "12.9", ... } }   <- step 1
        } } } }

Unlike the bash version (which echoes a JSON string that callers re-parse with
jq), this returns the resolved config as a dict — or None when nothing matches.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# The deployment targets bash understands (utils/set_properties.sh documents
# them, and packer/variables.pkr.hcl validates against the same list). They are
# also the keys used for node-type overrides in versions.json.
NODE_TYPES = ("azure_vm_regular", "azure_vm_akshost", "baremetal_1p", "baremetal_3p")

# What bash assumes when TARGET_NODE_TYPE is unset or empty.
DEFAULT_NODE_TYPE = "azure_vm_regular"

# Key prefixes that mark a node-type entry as a table of per-SKU overrides
# (e.g. "nvidia_v100", "amd_mi300"), rather than a config that applies to
# every SKU. Mirrors the startswith("nvidia_") / startswith("amd_") test in bash.
_SKU_KEY_PREFIXES = ("nvidia_", "amd_")

# Where installed component versions are recorded (mirrors the Bash
# write_component_version in utils/utilities.sh).
_VERSIONS_FILE = "/opt/azurehpc/component_versions.txt"


def normalize_key(value: str) -> str:
    """Mirror normalize_component_config_key: lowercase and collapse any run of
    non-alphanumeric characters into a single underscore."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower())


def _child(node, key):
    """Return node[key] if `node` is a dict, else None.

    Lets us walk a path such as versions["cuda"]["ubuntu24.04"]["x86_64"] one
    step at a time, getting None (instead of a KeyError or TypeError) as soon as
    any step is missing, which is how jq treats a missing path.
    """
    return node.get(key) if isinstance(node, dict) else None


def _is_sku_override_table(node) -> bool:
    """True if a node-type entry holds per-SKU overrides rather than a config.

    Mirrors bash's check: an object that has a "default" key, or any key starting
    with "nvidia_" or "amd_". Such a table must not be returned as-is, because
    its contents only apply to the SKUs it names. For example,
    hpcx.ubuntu24.04.x86_64.azure_vm_regular holds only {"nvidia_v100": {...}};
    an A100 must fall through to the architecture-level config instead.
    """
    if not isinstance(node, dict):
        return False
    # any(...) is True if at least one key starts with one of the prefixes.
    # str.startswith accepts a tuple and checks each prefix in it.
    return "default" in node or any(key.startswith(_SKU_KEY_PREFIXES) for key in node)


def get_component_config(component, versions, *, distribution, architecture,
                         gpu=None, sku=None, node_type=DEFAULT_NODE_TYPE):
    """Resolve the config for `component` from the `versions` dict.

    Follows the five-step lookup described at the top of this module, exactly
    as bash's get_component_config does. Returns the matching config (normally
    a dict), or None if no step matches.

    Every check below uses `is None` rather than a plain truth test, because bash
    only falls through when jq prints "null". A value such as {} or false counts
    as found in bash, so it must count as found here too.
    """
    arch_node = _child(_child(_child(versions, component), distribution), architecture)

    config = None

    # Steps 1-3 need both a vendor and a SKU, as in bash.
    if gpu and sku:
        sku_key = normalize_key(f"{gpu}_{sku}")
        # An empty node_type falls back to the default, like bash's
        # ${TARGET_NODE_TYPE:-azure_vm_regular}.
        node_type_key = normalize_key(node_type or DEFAULT_NODE_TYPE)
        node_type_node = _child(arch_node, node_type_key)

        # 1. <node_type>.<gpu_sku>: an override for this exact SKU and target.
        config = _child(node_type_node, sku_key)

        # 2. <node_type>.default: an override for every SKU on this target.
        if config is None:
            config = _child(node_type_node, "default")

        # 3. <node_type> itself, if it is a config rather than an override table.
        if (config is None and node_type_node is not None
                and not _is_sku_override_table(node_type_node)):
            config = node_type_node

    # 4. architecture level
    if config is None:
        config = arch_node

    # 5. common
    if config is None:
        config = _child(_child(versions, component), "common")

    return config


def config_for(component, env):
    """Resolve `component`'s config using the COMPONENT_VERSIONS + machine env
    vars that component_env() populates.

    A convenience wrapper around get_component_config for component modules,
    which all resolve their config from `env` the same way. The node type comes
    from TARGET_NODE_TYPE, the same variable bash reads.
    """
    versions = json.loads(env.get("COMPONENT_VERSIONS", "{}"))
    return get_component_config(
        component, versions,
        distribution=env.get("DISTRIBUTION", ""),
        architecture=env.get("ARCHITECTURE", ""),
        gpu=env.get("GPU"),
        sku=env.get("SKU"),
        node_type=env.get("TARGET_NODE_TYPE", DEFAULT_NODE_TYPE),
    )

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
