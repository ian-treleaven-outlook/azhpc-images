# Building images with the Python CLI

The Python CLI (`azhpc.py`) runs the same build as `distros/<os>/install.sh`,
step for step (see [porting-guide.md](porting-guide.md)). This page explains
how to build an image with it through Packer, and how to compare that image
with one built the usual bash way.

## What it covers today

| | Python CLI | Notes |
|---|---|---|
| Ubuntu 22.04, 24.04, 26.04 | yes | Ubuntu 26.04 refuses V100, GB200, VR200 and NCv6, as its install.sh does |
| Azure Linux 3.0 | yes | |
| AlmaLinux, Rocky, RHEL | no | Plans in ADO 995–997 (Packer already runs `azhpc.py` with a new enough Python there) |
| `azure_vm_regular` | yes | |
| `baremetal_3p` | yes | |
| `azure_vm_akshost` | no | Runs `install_aks.sh`, which has no Python plan |
| `baremetal_1p` | no | Needs the out-of-repo 1P overlay, which replaces bash files |
| SKUs | every SKU Packer passes | V100, A100, MI300X, GB200 (also GB300 VMs), VR200 (built as GB200 on Ubuntu 24.04), NCv6 |

Packer stops before creating a VM if `USE_PYTHON_CLI` is set for anything
else, with an `ERROR: use_python_cli is set, ...` message.

## How to switch a build to Python

Set `USE_PYTHON_CLI=true` (or `-var use_python_cli=true`). Everything else
stays as it is:

```bash
cd packer
export USE_PYTHON_CLI=true
packer build --var 'target_vm_size=Standard_ND96asr_v4' --var 'os_family=ubuntu' \
    --var 'distro_version=24.04' --var 'azure_location=southcentralus' \
    --on-error=run-cleanup-provisioner .
```

Only the "Install HPC components" step changes. Instead of
`bash install.sh <GPU> <SKU>` in `distros/<folder>/`, it runs

```bash
cd /home/<user>/azhpc-images; bash packer/scripts/run_python_cli.sh \
    --vendor <NVidia|AMD> --gpu <SKU> --os <Ubuntu22|Ubuntu24|Ubuntu26|Azure3>
```

`run_python_cli.sh` runs `azhpc.py` with a Python 3.10 or newer. Ubuntu and
Azure Linux have one as `python3`. AlmaLinux, Rocky and RHEL have 3.6 or 3.9,
so there it uses the newer Python their `install_utils.sh` installs later in
the build anyway (python3.12; python3.11 on Rocky 8), installing it first with
`dnf` if needed. The image gets no package a bash build wouldn't have, and
`python3` still points at the distro's own Python.

with the same environment (`TARGET_NODE_TYPE`, `NVLINK_RACKSCALE`,
`KERNEL_VERSION`, `REFRESH_MODE`). The prerequisites before it and the reboot,
tests, health checks and image capture after it are identical. Unset or
`false` builds exactly as before.

Python-built images get an extra tag, `AzhpcBuildCli = python`.

> **Don't publish comparison images as release candidates.** With
> `CREATE_IMAGE=true` the image goes to the `AzHPCImageReleaseCandidates`
> gallery by default. Use a test gallery or image definition (`sig_gallery_name`,
> `sig_image_name`), or leave `CREATE_IMAGE` unset.

## Reading the build log

`azhpc.py` writes one JSON object per line (it logs JSON when not attached to
a terminal), so the Packer output can be filtered with `jq`. Useful fields:

- the first `build-image` line: vendor, GPU, OS, distribution, architecture,
  node type and planned step count;
- per step: `starting (python)` / `done in Ns` with `runner` and `duration_s`;
  a failure is an ERROR with `exit_code` and the command's last 50 output lines;
- `WARNING` lines: bash quirks kept for parity, skipped steps that may hide an
  upstream gap, and steps that ran bash because their port is marked `resync`;
- the last `build-image` line: total duration and how many steps ran as
  Python, as bash, or were skipped.

Exit codes: 0 built, 1 invalid arguments, 2 bad `versions.json`, 3 a step
failed. Packer fails the build on any non-zero code, as it does for bash.

`azhpc.py --vendor NVidia --gpu A100 --os Ubuntu24 --dry-run` prints the plan
without building, and shows which steps run as Python and which as bash.

## Comparing a Python build with a bash build

Build a pair from the **same commit** (so `versions.json` is the same), with
the same variables, as close together in time as possible (both builds run
`apt upgrade`, and package mirrors move). The only difference should be
`USE_PYTHON_CLI`. Set `CAPTURE_IMAGE_FINGERPRINT=true` for **both** builds.

```bash
cd packer
export CAPTURE_IMAGE_FINGERPRINT=true
packer build ... .                                   # bash build
cp -r /tmp/image_manifests ~/compare/bash-a100
USE_PYTHON_CLI=true packer build ... .               # Python build
cp -r /tmp/image_manifests ~/compare/python-a100
python3 tools/compare_images.py ~/compare/bash-a100 ~/compare/python-a100
```

1. **Validation.** Packer runs `/opt/azurehpc/test/run-tests.sh` and, where
   the SKU supports it, the health checks, on both. Both must pass.
2. **What the images contain.** `tools/compare_images.py` compares everything
   the two manifest folders hold and prints, section by section, what is only
   in one build or differs between them:
   - from `build-evidence.tgz` (every build): installed packages and versions,
     package holds, repositories, service states, `dkms status`;
   - `component-versions.json` (every build; `ImageVersion` is ignored);
   - `trivy-report-rootfs.json` (every build): vulnerabilities found;
   - `image-fingerprint.json` (with `CAPTURE_IMAGE_FINGERPRINT`): every config
     file the build writes (modprobe, udev, sysctl, limits, profile.d, systemd
     units and drop-ins, ld.so.conf.d, apt/dnf config, waagent, netplan...)
     with its mode and SHA256, the layout of `/opt` and `/usr/local`, the
     modulefiles, enabled unit files and pip packages.

   Exit status 0 means no differences. `--ignore REGEX` skips items you have
   already explained; `--labels` renames the two builds in the report.
3. **Explain every difference.** A package version can differ when the two
   builds pulled from a mirror at different times; anything else is a parity
   bug to report against the step that writes it (the report names the path or
   package; the build log's `starting (python)` lines name the step).

The fingerprint step runs `tools/image_fingerprint.py` on the build VM after
validation (stdlib only, works with any `python3` the VM has, even Python 3.6),
downloads it next to the other manifests, and removes it from the VM, so the
image itself is unchanged. Known, deliberate differences are listed in each
Python module's docstring ("Deliberate differences") and in the porting guide.