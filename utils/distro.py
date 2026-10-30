"""distro.py — the build's distribution, as an object you can ask questions.

Bash scripts test `$DISTRIBUTION` (os-release ID + VERSION_ID, e.g.
"ubuntu24.04", "almalinux9.8") with string patterns, again and again:

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then ...
    elif [[ $DISTRIBUTION == almalinux* || $DISTRIBUTION == rocky* || ... ]]; then ...

A port asks a Distro instead, so the test is written once, here, and reads as
what it means:

    distro = Distro.from_env(env)
    if distro.is_ubuntu: ...
    elif distro.is_rhel_family: ...
    if distro.major == "9": ...

The answers match bash's patterns (e.g. is_ubuntu is `== *"ubuntu"*`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NAME_RE = re.compile(r"^([a-z]+)(.*)$")


@dataclass(frozen=True)
class Distro:
    """A distribution name such as "ubuntu24.04" (bash's $DISTRIBUTION)."""

    name: str

    @classmethod
    def from_env(cls, env: dict[str, str]) -> Distro:
        return cls(env.get("DISTRIBUTION", ""))

    @property
    def id(self) -> str:
        """The os-release ID part: "ubuntu", "almalinux", "azurelinux", "rocky", "rhel"."""
        match = _NAME_RE.match(self.name)
        return match.group(1) if match else ""

    @property
    def version(self) -> str:
        """The VERSION_ID part: "24.04", "9.8", "3.0"."""
        match = _NAME_RE.match(self.name)
        return match.group(2) if match else ""

    @property
    def major(self) -> str:
        """The major version: "24", "9", "3" (bash: ${VERSION_ID%%.*})."""
        return self.version.split(".")[0]

    @property
    def is_ubuntu(self) -> bool:
        return "ubuntu" in self.name

    @property
    def is_azurelinux(self) -> bool:
        return self.name.startswith("azurelinux")

    @property
    def is_almalinux(self) -> bool:
        return self.name.startswith("almalinux")

    @property
    def is_rocky(self) -> bool:
        return self.name.startswith("rocky")

    @property
    def is_rhel(self) -> bool:
        return self.name.startswith("rhel")

    @property
    def is_rhel_family(self) -> bool:
        """AlmaLinux, Rocky Linux or RHEL (not Azure Linux, which is RPM-based too)."""
        return self.is_almalinux or self.is_rocky or self.is_rhel

    @property
    def is_rpm(self) -> bool:
        """Any RPM-based distro this project builds: Azure Linux or the RHEL family."""
        return self.is_azurelinux or self.is_rhel_family

    @property
    def family(self) -> str:
        """"ubuntu", "azurelinux" or "rhel" (AlmaLinux, Rocky, RHEL); "" otherwise.

        For tables of per-distro data, e.g. _DEPS[distro.family].
        """
        if self.is_ubuntu:
            return "ubuntu"
        if self.is_azurelinux:
            return "azurelinux"
        if self.is_rhel_family:
            return "rhel"
        return ""

    def __str__(self) -> str:
        return self.name
