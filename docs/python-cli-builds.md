# Building images with the Python CLI

The Python CLI (`azhpc.py`) runs the same build as `distros/<os>/install.sh`,
step for step (see [porting-guide.md](porting-guide.md)). This page explains
how to build an image with it through Packer, and how to compare that image
with one built the usual bash way.

## What it covers today

| | Python CLI | Notes |
|---|---|---|
| Ubuntu 24.04 | yes | The only distro with a parity build plan so far. Others: ADO 991–997 |
| `azure_vm_regular` | yes | |
| `baremetal_3p` | yes | |
| `azure_vm_akshost` | no | Runs `install_aks.sh`, which has no Python plan |
| `baremetal_1p` | no | Needs the out-of-repo 1P overlay, which replaces bash files |
| SKUs | every SKU Packer passes | V100, A100, MI300X, GB200 (also GB300 VMs), VR200 (built as GB200), NCv6 |

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
`bash install.sh <GPU> <SKU>` in `distros/ubuntu24.04/`, it runs

```bash
cd /home/<user>/azhpc-images; python3 --version; \
    python3 azhpc.py --vendor <NVidia|AMD> --gpu <SKU> --os Ubuntu24
```

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
`USE_PYTHON_CLI`.

1. **Validation.** Packer runs `/opt/azurehpc/test/run-tests.sh` and, where
   the SKU supports it, the health checks, on both. Both must pass.
2. **Component versions.** Packer downloads
   `/opt/azurehpc/component_versions.txt` to
   `/tmp/image_manifests/component-versions.json`. Save it from each build and
   compare:

   ```bash
   diff <(jq -S . bash/component-versions.json) <(jq -S . python/component-versions.json)
   ```

   Expect only `ImageVersion` to differ.
3. **Vulnerability scan.** Compare the two `trivy-report-rootfs.json` reports
   from the same folder.
4. **The image itself.** Build with `RETAIN_VM_ALWAYS=true` and no artifacts,
   then run the same commands on both VMs and diff the output:

   ```bash
   dpkg-query -W -f '${Package} ${Version}\n' | sort         # packages
   apt-mark showhold                                          # held packages
   systemctl list-unit-files --state=enabled --no-pager       # enabled services
   ls -la /etc/modprobe.d /etc/udev/rules.d /etc/sysctl.d /etc/security/limits.d
   find /opt /usr/share/modules/modulefiles -maxdepth 2 | sort
   cat /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null
   ```

   Differences in timestamps, build folders and log files are expected; any
   other difference is a parity bug to report against the step that writes it.

Known, deliberate differences are listed in each Python module's docstring
("Deliberate differences") and in the porting guide.
