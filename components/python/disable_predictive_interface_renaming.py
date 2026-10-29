"""disable_predictive_interface_renaming.py — Python port of
distros/<distro>/disable_predictive_interface_renaming.sh.

What this step does
-------------------
Linux normally gives network interfaces "predictable" names such as enP1s0 or
ens5. The kernel parameter `net.ifnames=0` switches back to the classic names
(eth0, eth1, ...), which the HPC image and its tooling expect. This step adds
that parameter to the GRUB settings file used by Ubuntu cloud images,
/etc/default/grub.d/50-cloudimg-settings.cfg, and then runs `update-grub` so the
change goes into the generated boot menu (/boot/grub/grub.cfg).

The original bash, for comparison (the same file exists for ubuntu24.04 and
ubuntu22.04):

    set -e

    # Kernel Parameter to disable predictive network interface naming
    KERNEL_PARAMETER="net.ifnames=0"
    GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX console=tty1 console=ttyS0 earlyprintk=ttyS0"

    # Update 50-cloudimg-settings.cfg with additional kernel parameter
    sed -i "s/${GRUB_CMDLINE_LINUX}/${GRUB_CMDLINE_LINUX} ${KERNEL_PARAMETER}/" /etc/default/grub.d/50-cloudimg-settings.cfg
    # Generate grub file with updated parameters
    update-grub

How the sed works, step by step:

  * GRUB_CMDLINE_LINUX starts from whatever value the build environment has.
    Normally it isn't set, so the text to search for becomes
        " console=tty1 console=ttyS0 earlyprintk=ttyS0"
    Note the leading SPACE: it comes from the space between the (empty)
    "$GRUB_CMDLINE_LINUX" and "console=tty1".
  * On each line of the file, sed replaces the first occurrence of that text
    with the same text followed by " net.ifnames=0".

Because of the leading space, the search only matches when something comes
before "console=tty1" on the line. A real GNU sed run in WSL shows:

    GRUB_CMDLINE_LINUX_DEFAULT="console=tty1 console=ttyS0 earlyprintk=ttyS0"
        -> unchanged (a quote, not a space, comes before "console=tty1")
    GRUB_CMDLINE_LINUX="quiet console=tty1 console=ttyS0 earlyprintk=ttyS0"
        -> GRUB_CMDLINE_LINUX="quiet console=tty1 console=ttyS0 earlyprintk=ttyS0 net.ifnames=0"

This port reproduces that exactly, including the "unchanged" case (see
"Flagged bash behavior" below). Running the step twice adds net.ifnames=0 a
second time, again exactly like bash.

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"disable-predictive-interface-renaming" Step. When a Step has an action, the
build calls the Python function instead of running the bash script. Deleting
`action=` from that Step makes the build run the bash script again. That is why
the .sh file is kept.

Like every action, install() receives the build environment (a dict of
environment variables) and returns 0 for success or 3 for failure.

Ubuntu only
-----------
build_plan runs this step only on Ubuntu, as install.sh does; on other distros
it is skipped with a warning. (If it were run elsewhere anyway, the Ubuntu
cloud-image file /etc/default/grub.d/50-cloudimg-settings.cfg would be missing,
so install() would log an error and return 3 before running update-grub.)

Flagged bash behavior (mirrored, not fixed)
-------------------------------------------
If the image's 50-cloudimg-settings.cfg has the console parameters at the very
start of the quoted value (as in the GRUB_CMDLINE_LINUX_DEFAULT example above),
the search text never matches, so net.ifnames=0 is not added and the step
still succeeds. Whether that happens depends on the base image's file, which is
not in this repository. install() logs a warning (with the file and the search
text) whenever no line matched, so a build log shows it.

Deliberate differences
----------------------
None for normal values. sed treats the search text as a regular expression;
this port matches it literally. With the normal value (letters, digits, `=`,
and spaces) there is no difference. A GRUB_CMDLINE_LINUX value containing
regex characters such as `.` or `*`, a `/` (which breaks the sed command), or
`&` (special in sed's replacement) could behave differently in sed.
"""

from __future__ import annotations

from utils.logger import log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in build_config.py. Every log line from this module uses it.
_OP = "disable-predictive-interface-renaming"

# The GRUB settings file that Ubuntu cloud images ship, edited by bash's sed.
_GRUB_SETTINGS_FILE = "/etc/default/grub.d/50-cloudimg-settings.cfg"

# Same values as the bash variables.
_KERNEL_PARAMETER = "net.ifnames=0"
_CONSOLE_PARAMETERS = "console=tty1 console=ttyS0 earlyprintk=ttyS0"


def search_text(env: dict[str, str]) -> str:
    """Build the text that bash's sed searches for.

    Mirrors bash's
        GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX console=tty1 console=ttyS0 earlyprintk=ttyS0"

    An unset variable expands to "" in bash, so env.get(..., "") gives the same
    result. With the usual unset value the result starts with a space:
        " console=tty1 console=ttyS0 earlyprintk=ttyS0"
    """
    return f"{env.get('GRUB_CMDLINE_LINUX', '')} {_CONSOLE_PARAMETERS}"


def add_kernel_parameter(text: str, search: str, parameter: str) -> str:
    """Return `text` with " <parameter>" added after `search` on each line.

    Mirrors `sed "s/<search>/<search> <parameter>/"`: on every line, only the
    first occurrence of `search` is changed, and lines without it are left as
    they are. Running it again adds the parameter again, as sed would.

    Note: sed treats `search` as a regular expression, while this function
    matches it literally. They agree for the normal value, which contains no
    regex-special characters; see the module docstring.

    This is a "pure" function: it only transforms its input, so tests can call
    it directly with real file contents.
    """
    replacement = f"{search} {parameter}"
    # sed works line by line, and only "\n" ends a line. Splitting on "\n" and
    # joining back keeps the file's final newline (or lack of one) unchanged.
    # str.replace(old, new, 1) replaces only the first occurrence, like sed's
    # s/// without the `g` flag.
    return "\n".join(line.replace(search, replacement, 1)
                     for line in text.split("\n"))


def _update_settings_file(search: str) -> bool:
    """Apply add_kernel_parameter() to the GRUB settings file, like `sed -i`.

    Returns True on success, or False after logging why it failed (for example,
    the file does not exist, which also makes `sed -i` fail).
    """
    # newline="" turns off Python's newline translation, so "\r\n" stays as it
    # is. errors="surrogateescape" lets bytes that aren't valid UTF-8 pass
    # through unchanged, since sed doesn't care about the encoding either.
    try:
        with open(_GRUB_SETTINGS_FILE, encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            original = f.read()
        updated = add_kernel_parameter(original, search, _KERNEL_PARAMETER)
        with open(_GRUB_SETTINGS_FILE, "w", encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            f.write(updated)
    except OSError as exc:
        log_error(_OP, f"could not update {_GRUB_SETTINGS_FILE}: {exc}")
        return False

    if updated == original:
        # Not an error: bash's sed also succeeds when nothing matches. But it
        # means net.ifnames=0 was NOT added, so interfaces keep their
        # predictable names - worth a warning in the build log.
        log_warn(_OP, f"No line in {_GRUB_SETTINGS_FILE} contains the search "
                      f"text, so {_KERNEL_PARAMETER} was not added (bash's sed "
                      "succeeds silently here too)",
                 path=_GRUB_SETTINGS_FILE, search=search, lines_changed=0)
    else:
        # Every line that contains the search text is changed exactly once.
        changed = sum(1 for line in original.split("\n") if search in line)
        log_info(_OP, f"Added {_KERNEL_PARAMETER} in {_GRUB_SETTINGS_FILE}",
                 path=_GRUB_SETTINGS_FILE, lines_changed=changed)
    return True


def install(env: dict[str, str]) -> int:
    """Add net.ifnames=0 to the GRUB settings and regenerate the GRUB config.

    `env` is the build environment. GRUB_CMDLINE_LINUX is read from it (usually
    unset), and it is passed on to update-grub.
    Returns 0 on success and 3 on failure.
    """
    # ---- 1. edit the settings file (the sed line) --------------------------
    if not _update_settings_file(search_text(env)):
        return 3

    # ---- 2. regenerate /boot/grub/grub.cfg ---------------------------------
    if exec_program(["update-grub"], _OP, env=env) != 0:
        log_error(_OP, "update-grub failed")
        return 3

    log_info(_OP, "Regenerated the GRUB configuration")
    return 0
