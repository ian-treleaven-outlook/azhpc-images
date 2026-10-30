"""plans — one build plan per distro, mirroring distros/<folder>/install.sh.

Every distro's install.sh runs a different list of component scripts, in a
different order, with different conditions (Azure Linux uses MOFED instead
of DOCA; AlmaLinux, Rocky and RHEL add fix_setools_cyclecloud.sh and
network-config.sh; Ubuntu 22.04 runs remove_unused_packages.sh before
install_utils.sh; ...). So each distro gets its own plan module, written to
be read side by side with its install.sh, and built from the shared step
library in steps.py.

How a build runs (ImageBuilder in utils/build_config.py):

    1. set_properties.sh's system changes       utils/system_prep.py
    2. the plan's pre-bootstrap steps           DistroPlan.pre_bootstrap
    3. install_utils.sh                         (bash)
    4. the plan's steps                         DistroPlan.build

Adding a distro (ADO 992-997): write utils/plans/<folder>.py with
`build(cfg)` (and `pre_bootstrap(cfg)` if its install.sh runs something
before install_utils.sh), register it in PLANS, drop it from PENDING, and add
its folder to python_cli_os_by_folder in packer/variables.pkr.hcl. The parity
tests in tests/python/test_distro_plans.py then check it against its
install.sh automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from utils.plans import almalinux, azurelinux3_0, rocky, ubuntu22_04, ubuntu24_04, ubuntu26_04

if TYPE_CHECKING:
    from utils.build_config import BuildConfig
    from utils.plans.step import Step


@dataclass(frozen=True)
class TargetOS:
    """An --os value: its distros/ folder, and the distribution it builds.

    `distribution` is the os-release ID+VERSION_ID that versions.json is
    keyed on for the current release (e.g. almalinux9.8 for the almalinux9.x
    folder). `azhpc.py versions` and `validate` use it; a real build reads
    the build VM's /etc/os-release instead, as bash does.
    """
    folder: str
    distribution: str


OS_TARGETS: dict[str, TargetOS] = {
    "Ubuntu22": TargetOS("ubuntu22.04", "ubuntu22.04"),
    "Ubuntu24": TargetOS("ubuntu24.04", "ubuntu24.04"),
    "Ubuntu26": TargetOS("ubuntu26.04", "ubuntu26.04"),
    "Azure3":   TargetOS("azurelinux3.0", "azurelinux3.0"),
    "Alma8":    TargetOS("almalinux8.10", "almalinux8.10"),
    "Alma9":    TargetOS("almalinux9.x", "almalinux9.8"),
    "Rocky8":   TargetOS("rocky8.10", "rocky8.10"),
    "Rocky9":   TargetOS("rocky9.x", "rocky9.8"),
    "RHEL8":    TargetOS("rhel8.10", "rhel8.10"),
    "RHEL9":    TargetOS("rhel9.x", "rhel9.8"),
}


def _no_steps(cfg: BuildConfig) -> list[Step]:
    return []


def _accept_every_sku(sku: str) -> str:
    return ""


@dataclass(frozen=True)
class DistroPlan:
    """How to build one distro, mirroring its install.sh."""

    # The steps after install_utils.sh, in install.sh's order.
    build: Callable[[BuildConfig], list[Step]]
    # Steps install.sh runs before install_utils.sh (most distros: none).
    pre_bootstrap: Callable[[BuildConfig], list[Step]] = _no_steps
    # The vendors install.sh accepts. e.g. AlmaLinux's install.sh exits with
    # "AMD GPU support is not implemented yet for AlmaLinux."
    vendors: frozenset[str] = field(default_factory=lambda: frozenset({"NVidia", "AMD"}))
    # install.sh's message for a vendor it rejects.
    vendor_error: str = ""
    # SKUs install.sh builds as another SKU, e.g. Ubuntu 24.04's
    #     export SKU=$([[ $2 == "VR200" ]] && echo "GB200" || echo "$2")
    sku_aliases: dict[str, str] = field(default_factory=dict)
    # install.sh's message for a SKU it refuses outright, or "" to build it.
    reject_sku: Callable[[str], str] = _accept_every_sku


# The --os values with a Python build plan.
PLANS: dict[str, DistroPlan] = {
    "Ubuntu22": DistroPlan(build=ubuntu22_04.build, pre_bootstrap=ubuntu22_04.pre_bootstrap),
    "Ubuntu24": DistroPlan(build=ubuntu24_04.build, sku_aliases={"VR200": "GB200"}),
    "Ubuntu26": DistroPlan(build=ubuntu26_04.build, reject_sku=ubuntu26_04.reject_sku),
    "Azure3": DistroPlan(build=azurelinux3_0.build),
    "Alma8": DistroPlan(build=almalinux.build, vendors=frozenset({"NVidia"}),
                        vendor_error=almalinux.VENDOR_ERROR),
    "Alma9": DistroPlan(build=almalinux.build, vendors=frozenset({"NVidia"}),
                        vendor_error=almalinux.VENDOR_ERROR),
    "Rocky8": DistroPlan(build=rocky.build),
    "Rocky9": DistroPlan(build=rocky.build),
}

# The --os values still waiting for one, and the ticket adding it.
PENDING: dict[str, str] = {
    "RHEL8": "ADO 997",
    "RHEL9": "ADO 997",
}
