"""version_report.py — resolve every component's version for a target config.

Step 1 of version-compatibility validation: walk versions.json and produce the
flat list of component -> version that *would* be installed for a given
distribution / architecture / gpu / sku / node type. It resolves nothing itself
beyond reusing get_component_config, so the answers match what the build uses.

This module only gathers and reports versions; it does not judge whether a
combination is valid (that is Step 2).
"""

from __future__ import annotations

from utils.component_config import get_component_config

# Top-level keys in versions.json that are metadata, not installable components.
_NON_COMPONENTS = {"note"}


def _extract_versions(config):
    """Pull the version string(s) out of a resolved config dict.

    Most components expose a flat "version". A few (e.g. cuda) have no top-level
    version and instead nest versions under sub-keys such as driver/samples.
    Returns a list of (sub_label, version) pairs; sub_label is "" for the flat
    case.
    """
    if not isinstance(config, dict):
        return []
    # A flat top-level version is the one that would actually be used; do not
    # recurse into more-specific overrides that only apply to other SKUs.
    if isinstance(config.get("version"), str):
        return [("", config["version"])]
    # No flat version: collect versions from immediate sub-dicts (cuda case).
    pairs = []
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(value.get("version"), str):
            pairs.append((key, value["version"]))
    return pairs


def resolve_report(versions, *, distribution, architecture,
                   gpu=None, sku=None, node_type="azure-vm"):
    """Resolve every component's version for the given target.

    Returns a list of dicts, one row per (component[.sub]) entry:
        {"component": str, "version": str | None, "resolved": bool}
    - resolved=False  -> the component exists but no config matched the target.
    - version=None    -> a config matched but carried no version string.
    """
    report = []
    for component in sorted(versions):
        if component in _NON_COMPONENTS:
            continue
        config = get_component_config(
            component, versions,
            distribution=distribution, architecture=architecture,
            gpu=gpu, sku=sku, node_type=node_type,
        )
        if config is None:
            report.append({"component": component, "version": None, "resolved": False})
            continue
        pairs = _extract_versions(config)
        if not pairs:
            report.append({"component": component, "version": None, "resolved": True})
            continue
        for sub, version in pairs:
            label = component if not sub else f"{component}.{sub}"
            report.append({"component": label, "version": version, "resolved": True})
    return report


def format_report(report):
    """Render a report (from resolve_report) as an aligned text table."""
    width = max((len(row["component"]) for row in report), default=0)
    lines = []
    for row in report:
        if not row["resolved"]:
            version = "(no match)"
        elif row["version"] is None:
            version = "(no version)"
        else:
            version = row["version"]
        lines.append(f"{row['component']:<{width}}  {version}")
    return "\n".join(lines)
