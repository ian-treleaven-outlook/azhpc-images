# Porting the azhpc build to Python

A contributor guide for the incremental port of the bash image-build
orchestration to Python.

**Status:** [`build_plan()`](../utils/build_config.py#L191) in
[`build_config.py`](../utils/build_config.py#L191), called from
[`azhpc.py`](../azhpc.py#L192), has 34 steps; **all 34 run natively in Python**,
matching upstream `Azure/azhpc-images@0c2b8f9` (October 2026; see §12).
Each bash script is kept as that step's fallback and parity reference: delete a
Step's `action=`, or mark it `resync=`, and it runs the bash script again. The
plan mirrors `distros/ubuntu24.04/install.sh`; the other distros' `install.sh`
scripts still differ from it in places (see §12).

> Much of the design rationale here comes from Gio Martinez's project
> presentation, *Azure HPC Images Python Refactor* (Aug 2026). He designed and
> built the foundation this guide describes.

---

## 1. Why this refactor exists

The bash foundation works and has for years. The problem isn't that any one
script is bad — it's that **the same logic exists in many copies, and nothing
keeps them in sync.** Three specific problems motivated the port.

### Problem 1 — maintenance: distro branching copied everywhere

Almost every component contains a variant of this:

```bash
# components/install_nccl.sh
if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
    apt install -y build-essential devscripts debhelper fakeroot zlib1g-dev libibverbs-dev
elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
    tdnf install -y rpm-build rpmdevtools autoconf automake git libtool
else
    yum install -y rpm-build rpmdevtools
fi
```

**27 of 45 component scripts branch on `$DISTRIBUTION`** this way, each edited
independently.

That one `if` is doing *three* jobs at once: choosing a package manager, listing
the packages, and running the install. Only the package list is genuinely
specific to NCCL. And because it matches the distro **by name as a string**, a
new Azure Linux version falls through to `else` and runs `yum` on a machine that
doesn't have it — which you discover hours into a build.

### Problem 2 — configuration: re-parsed and unchecked

```bash
nccl_metadata=$(get_component_config "nccl")
NCCL_VERSION=$(jq -r '.version' <<< $nccl_metadata)
COMMIT=$(jq -r '.rdmasharpplugins.commit' <<< $nccl_metadata)
```

Every field re-invokes `jq` on the same string — **112 `jq` calls across
`components/`, 13 in `install_mpis.sh` alone.** Worse, none are checked:
`jq -r` on a missing key returns an *empty string*, not an error. The version
comes back blank, the build continues, and you get a download URL with a hole
in it.

### Problem 3 — logging: `set -x` and nothing else

Raw shell tracing streams every executed command with no structure, no step
labels, and — notably — never shows the exit code. If a build failed two hours
in, that's what you scrolled through. It's only readable if you already know
what every line was supposed to do.

---

## 2. How the port works

`distros/<os>/install.sh` is the authoritative bash orchestrator.
`utils/build_config.py` mirrors it as a **build plan** — an ordered list of
`Step` objects:

```python
@dataclass(frozen=True)
class Step:
    op: str                       # log label, e.g. "install-rocm"
    script: str = ""              # canonical bash script name (parity + bash exec)
    action: Callable[[dict[str, str]], int] | None = None  # optional python action
    args: tuple[str, ...] = ()
    base: str = "component"       # "component" (components/) or "distro" (build_dir)
    when: Callable[[BuildConfig], bool] = lambda cfg: True  # condition
    skip_warning: str = ""        # why a skip deserves a WARNING (see §7)
    resync: str = ""              # ticket, when the port is behind its bash
```

Every step keeps its bash script name, so the plan stays traceable and can be
diffed against `install.sh` automatically. **Bash is the default.** A step only
runs Python when it has an `action` and is not marked `resync`
(`Step.runner`):

```python
if step.runner == "python":
    rc = step.action(env)          # ported: run Python
else:
    rc = exec_program([...])       # not yet ported, or stale: run the bash script
```

### Porting is one reversible line

```python
Step("install-lustre", "install_lustre_client.sh"),                    # bash

Step("install-mpifileutils", "install_mpifileutils.sh",
     action=install_mpifileutils.install),                             # ported
```

Adding `action=` ports it. **Deleting `action=` reverts it** — back to code
that's worked for years, with no untangling and no large revert. That property
is the whole migration strategy: nothing is forced, and every component is
verified before moving to the next.

You are not rewriting the orchestrator, and you must **not** remove `script=` —
the parity tests depend on it.

### When upstream changes a ported script: `resync=`

Upstream keeps changing the bash. When a ported script changes, the port is
stale, and running it would build something different from bash. Mark the step
with the ticket that will update the port:

```python
Step("install-dcgm", "install_dcgm.sh", action=install_dcgm.install,
     when=nvidia, resync="ADO 987"),                                  # stale
```

The step then runs its bash script, logs a WARNING naming the ticket, and
`azhpc.py --dry-run` shows `(bash, resync ADO 987)`. The port stays wired, so
its tests keep running. To re-enable it, bring the port up to date with
`git diff <old> <new> -- components/<name>.sh` and delete `resync=`.
`ResyncTests` in `tests/python/test_build_plan.py` pins the list of marked steps,
so both directions are deliberate changes.

---

## 3. The action contract

```python
def install(env: dict[str, str]) -> int:
```

Four rules, all load-bearing:

1. **Return `0` on success, `3` on failure.** These are the exit codes
   `azhpc.py --help` publishes. Don't invent new ones.
2. **Don't let exceptions escape.** `build()` catches them and converts to
   `rc=3`, but that's a backstop, not the contract. A caught-and-logged failure
   is diagnosable; a traceback is not.
3. **Log through `utils.logger`, never `print()`.** Pass your step's `op` label
   first so output correlates with the step.
4. **Read configuration from `env`.** It already carries `COMPONENT_VERSIONS`
   (the contents of `versions.json`), `DISTRIBUTION`, `ARCHITECTURE`, `GPU`,
   `SKU`, `TARGET_NODE_TYPE`, and the `*_DIR` paths — see `component_env()`.

---

## 4. The toolkit

Everything you need exists. Use these rather than rolling your own.

| Helper | Import from | Signature |
|---|---|---|
| Resolve a component's config | `utils.component_config` | `config_for(component, env)` |
| Record installed version | `utils.component_config` | `write_component_version(component, version)` |
| Download + SHA256 verify | `utils.download` | `download_and_verify(url, sha256, dest_dir=".") -> Path` |
| Run an external command | `utils.process` | `exec_program(command: list[str], op, *, cwd=None, env=None) -> int` |
| Run and capture stdout | `utils.process` | `run_capture(command: list[str], op, *, cwd=None, env=None) -> tuple[int, str]` |
| Install distro packages | `utils.package_installer` | `PackageInstaller().install_package(packages) -> bool` |
| Pin packages against dnf updates (bash `dnf_pin_packages`) | `utils.package_pinning` | `dnf_pin_packages(packages, op, env=None) -> bool` |
| SKU network mode / IPoIB (bash `sku_network_mode`, `sku_uses_ipoib`) | `utils.sku` | `sku_network_mode(sku, node_type)`, `sku_uses_ipoib(sku, node_type)`, `sku_uses_ucx(sku)` |
| Logging | `utils.logger` | `log_info(op, msg)` · `log_warn(op, msg)` · `log_error(op, msg)` · `log_debug(op, msg)` · `log_error_detail(op, msg, detail)` |

---

## 5. Configuration: the five-tier version hierarchy

`config_for()` resolves a component by walking `versions.json` **most-specific
first**, in the same order as bash's `get_component_config`:

```
1. component.distribution.architecture.<node_type>.<gpu_sku>
2. component.distribution.architecture.<node_type>.default
3. component.distribution.architecture.<node_type>   (if it is a config, not a
                                                      table of per-SKU overrides)
4. component.distribution.architecture
5. component.common
```

Steps 1–3 only apply when both `GPU` and `SKU` are set. `<node_type>` is the
deployment target from `TARGET_NODE_TYPE` — `azure_vm_regular` (default),
`azure_vm_akshost`, `baremetal_1p` or `baremetal_3p`. `<gpu_sku>` is the vendor
and SKU normalised, e.g. `nvidia_v100`.

A real example — NCCL on Ubuntu 24.04, three machines, one file:

| Machine | Step hit | Version |
|---|---|---|
| GB200 · aarch64 · `baremetal_3p` | 3 | `2.28.3-1` |
| GB200 · aarch64 · `azure_vm_regular` | 4 | `2.29.7-1` |
| A100 · x86_64 · `azure_vm_regular` | 5 | `2.30.4-1` |

Each machine gets the most specific entry that exists for it. Step 1 is for a
single SKU on a single target: CUDA on Ubuntu 24.04 x86_64 is `13.0`, but a V100
on a regular VM gets `12.9` from `cuda…x86_64.azure_vm_regular.nvidia_v100`.

**This is the single most important function in the project** — every component
depends on it. Which is exactly why it has a differential test (§9).

Two rules follow:

- **Never hand-parse `versions.json`.** Call `config_for`.
- **Always check the result.** It returns `None` on a miss — not an empty
  string. That check is the fix for Problem 2:

```python
cfg = config_for("nccl", env)          # parse once
if not cfg or not cfg.get("version"):  # explicit miss, not a silent blank
    log_error("install-nccl", "could not resolve nccl version")
    return 3
```

---

## 6. Package management: three jobs, separated

The tangled `if/elif` from Problem 1 decomposes into three independently
testable pieces.

**1 — What the component needs is data, and lives with the component:**

```python
_DEPS = {
    "apt-get": ["build-essential", "zlib1g-dev", "libibverbs-dev"],
    "tdnf":    ["rpm-build", "rpmdevtools", "autoconf", "libtool"],
    "yum":     ["rpm-build", "rpmdevtools"],
}
```

The lists differ deliberately — `build-essential` doesn't exist on RHEL, and
Azure Linux needs packages RHEL doesn't. Note this table *names* package
managers but never runs one, and contains no distro check.

**2 — Which manager is a single shared function.** It walks `PATH` and finds the
first manager actually **installed**, in a fixed order:

```python
_MANAGER_ORDER = ["apt-get", "apt", "dnf", "tdnf", "yum", "zypper", "pacman", "apk"]
```

Order matters: RHEL has both `yum` and `dnf`, and `dnf` wins every time. A new
Azure Linux version needs **zero changes** — it finds `tdnf` and carries on,
instead of falling through to a `yum` that isn't there.

**3 — One execution gateway** runs the command, logs it, and returns the code:

```python
rc = exec_program(manager.install_command(package), "install-package")
if rc != 0:
    log_warn("install-package", f"failed (exit {rc})")
```

**So: don't write distro `if/elif` chains.** Declare a manager-keyed dict and
hand it to `PackageInstaller`.

---

## 7. Logging

Because every command goes through `exec_program`, every log line carries an
operation label. Two levels matter:

- **INFO** — one clear line per step: what started, what finished, how long.
  This is what you watch during a build. It answers *what happened*.
- **DEBUG** — every command's full output **plus the exit code** the old `set -x`
  trace never showed. It answers *why*.

The logger configures itself from the environment — no code changes to switch
behavior:

| Variable | Values |
|---|---|
| `LOG_FORMAT` | `text` \| `json` (text on a TTY, json otherwise) |
| `LOG_LEVEL` | `debug` \| `info` \| `warn` \| `error` |
| `LOG_FILE` | path (default: timestamped under `LOG_DIR`) |
| `RUN_ID` | ADO `BUILD_BUILDID`, else a UUID |

```json
{"ts":"...","level":"info","op":"install-doca","msg":"...",
 "run_id":"...","vendor":"NVidia","gpu":"A100"}
```

In a pipeline it emits JSON automatically, and `run_id` is the ADO build ID — so
a log line traces straight back to the build that produced it. Every line also
carries vendor/GPU/OS for filtering across concurrent builds.

### What the framework logs for you

- **Every command** (`exec_program` / `run_capture`): `Running command: ...` at
  INFO, its output at DEBUG, then one closing line with `exit_code` and
  `duration_s`. On failure that line is an ERROR whose detail block holds the
  command's **last 50 output lines**, so the real cause (e.g.
  `E: Unable to locate package foo`) is visible at the default INFO level.
- **Every step** (`build()`): `starting (python|bash)`, then `done in Ns` or the
  failure, with `runner`, `exit_code` and `duration_s` fields. An unexpected
  exception is logged with its traceback as the detail block. A step marked
  `resync` first logs a WARNING that its Python port is behind the bash script,
  with a `resync` field naming the ticket.
- **The run**: a first line with vendor, GPU, OS, distribution, architecture,
  node type and planned step count; a last line with total duration and how
  many steps ran as Python, as bash, or were skipped.

### Structured fields

Every log helper takes optional keyword fields:

```python
log_info(_OP, "Recorded version", component="PMIX", version=version)
```

JSON puts them under `"data"` (so a log query can filter on them); text appends
`[component=PMIX version=4.2.9]`. Values are redacted like messages.

### What a port should log

The aim: from the INFO log alone, someone can tell what a step did to the
image, and why it chose to.

- **Start:** what is being installed and the resolved inputs —
  `log_info(_OP, "Installing X", version=..., distro=..., source=url)`.
- **Decisions:** which branch was taken and why (distro handler, vendor, a
  skipped optional part), and any rerun/parity quirk worth a `log_warn`.
- **Changes to the system:** each file written or edited — path, and whether it
  was created, appended to or replaced — plus services enabled/started,
  symlinks, packages pinned or held.
- **Results:** every `write_component_version` value, and a closing
  `log_info(_OP, "X installed", ...)` summary.
- **Failures:** `log_error` with what failed and the values involved (path,
  command, expected vs found). Don't repeat a command failure that
  `exec_program` has already logged; add the *consequence* instead.
- Use `log_debug` for bulky detail (parsed listings, full file contents).
  Never log secrets; redaction is a safety net, not a licence.

---

## 8. Worked example

`components/python/install_intel_libs.py` is the cleanest end-to-end example —
read it first.

```python
def install(env: dict[str, str]) -> int:
    # 1. resolve config; bail cleanly if the component isn't in versions.json
    cfg = config_for("intel_one_mkl", env)
    if not cfg or not cfg.get("version"):
        log_error("install-intel-libs",
                  "could not resolve intel_one_mkl version from versions.json")
        return 3
    version = cfg["version"]

    log_info("install-intel-libs", f"Installing Intel oneAPI MKL {version}")

    # 2. download + verify — always guarded
    try:
        installer = download_and_verify(cfg.get("url", ""), cfg.get("sha256", ""),
                                        dest_dir="/tmp")
    except Exception as exc:
        log_error("install-intel-libs", f"download/verify failed: {exc}")
        return 3

    # 3. run the vendor installer; check the return code explicitly
    rc = exec_program([...], "install-intel-libs", env=env)
    if rc != 0:
        log_error("install-intel-libs", f"installer failed with exit code {rc}")
        return 3

    # 4. record what we installed
    write_component_version("INTEL_ONE_MKL", version)
    return 0
```

That shape — *resolve → download → execute → record → clean up* — covers most
remaining scripts.

**When yours is harder, study these:**

| Situation | Study |
|---|---|
| Needs distro packages first | `install_mpifileutils.py` + `_MPIFILEUTILS_DEPS` in `build_config.py` |
| Shell that must share one shell (e.g. `module load` then build) | `install_mpifileutils.py` — the build step |
| Three-way distro branching | `install_doca.py` |
| Needs stdout parsed from a command | `install_nccl.py` / `run_capture` |
| Inline shell in `install.sh` with no component script | `_cleanup_downloads` in `build_config.py` |

---

## 9. How to port a script

1. **Claim the ADO task** so nobody duplicates work.
2. **Read the bash** end to end. Note every external command, package install,
   downloaded file, and version written.
3. **Create `components/python/<name>.py`** with a docstring naming the bash
   script it ports, and an `install(env) -> int`.
4. **Attach the action** in `build_plan()` — add `action=<module>.install` to the
   existing `Step`. Keep `script=` and `when=` exactly as they are.
5. **Import your module** at the top of `utils/build_config.py`.
6. **Write tests** (below).
7. **Run the suite:** `python -m unittest discover -s tests/python`
8. **Open a PR** against `modern/python`.

### Leave the bash script in place

Do **not** delete `components/<name>.sh`. It remains the fallback for unported
distro paths, the script `install.sh` invokes, and the reference the parity
tests compare against. Scripts leave only when *upstream* deletes them:
`install_cmake.sh` earlier, and in the October 2026 sync
`install_dynolog_drl.sh`, `install_azure_persistent_rdma_naming_monitor.sh` and
the Ubuntu 22.04/24.04 `disable_auto_upgrade.sh`.

---

## 10. Testing expectations

Every port needs tests. This isn't ceremony: the parity suite is what caught 54
commits of upstream drift during the last rebase.

The suite is **1066 tests, ~standard library only** — `unittest` and
`unittest.mock`, no pytest. A few compare against real bash, git, jq or sed in
the test environment and skip themselves when those tools are missing.

Add coverage for at minimum:

- **happy path** — mock `download_and_verify` / `exec_program`; assert `install()`
  returns `0` and `write_component_version` was called;
- **missing version** — returns `3` and logs an error;
- **failing external command** — non-zero `rc` propagates as `3`.

Mock at the boundary: patch the helper on *your* module
(`mock.patch.object(install_foo, "download_and_verify", ...)`), never real
network or filesystem.

### The differential test

`tests/python/test_component_config.py` is the pattern worth understanding. It
doesn't hardcode what we *think* bash returns — it **executes the real bash
`get_component_config`** and diffs Python against it, for every component in
`versions.json` across four scenarios:

```python
for scenario in SCENARIOS.values():
    for component in self.versions:
        expected = bash_get_component_config(component, scenario)
        actual   = get_component_config(component, self.versions, **scenario)
        self.assertEqual(actual, expected)
```

It regenerates from the current `versions.json` on every run, so it can't go
stale. If your component has interesting resolution (SKU-nested or
`baremetal_3p` overrides), add a scenario here.

---

## 11. Gotchas

- **`module load` needs one shell.** `module` is a shell function; separate
  `exec_program` calls each get a fresh shell where it isn't defined. Pass the
  whole sequence to `bash -c`.
- **Don't add distro `if/elif` chains.** See §6.
- **`apt` is already non-interactive** — `component_env()` sets
  `DEBIAN_FRONTEND=noninteractive`. Don't re-set it or add `-y` hacks.
- **Order matters.** Some steps depend on earlier ones (`install_lustre_client.sh`
  must precede `install_mpifileutils.sh` so Lustre support compiles in). Don't
  reorder `build_plan()` to suit a port.
- **Log labels must match the step's `op`**, or output can't be correlated.

---

## 12. Status — every step ported, in line with upstream `0c2b8f9`

The October 2026 sync (ADO 983; upstream `f58ab80` → `0c2b8f9`) changed the
plan:

- the NVIDIA driver (GRID on NCv6) now runs **before** PMIx/MPI, so HPC-X can
  rebuild Open MPI with CUDA support;
- `install_dynolog_drl.sh` and the Ubuntu 22.04/24.04 `disable_auto_upgrade.sh`
  were removed upstream, so their steps are gone. The dynolog port was deleted
  (ADO 989); `disable_auto_upgrade.py` is kept for Ubuntu 26.04, which still
  runs that script;
- `install_azsecpack_prereqs.sh` is new, after `write_kernel_os_version.sh`
  (ported in ADO 990).

Upstream also changed 16 ported scripts. They were marked `resync=` during the
sync, and resynced with the new bash in:

| Ticket | Steps |
|---|---|
| ADO 984 | `install-pmix`, `install-mpis`, `install-mpifileutils` |
| ADO 985 | `install-nv-driver`, `install-nv-grid`, `install-nvloom`, `install-nvbandwidth` |
| ADO 986 | `install-rocm`, `install-rccl` |
| ADO 987 | `install-doca`, `install-nccl`, `install-dcgm` |
| ADO 988 | `install-lustre`, `hpc-tuning`, `install-persistent-rdma-naming`, `install-aznfs` |

The resync also brought the smaller Ubuntu 26.04 changes into the docker and
monitoring-tools ports, and fixed parity gaps found on the way (nvbandwidth,
Moneo, mpifileutils, AZNFS). Each port mirrors every distro branch of its bash,
including Ubuntu 26.04 and RHEL, ready for the per-distro plans.

The bash tree matches upstream exactly; the only differences are Python test
fixtures.

Nested component scripts that a ported step calls (e.g. `install_gdrcopy.sh`,
`install_nvidia_fabric_manager.sh`, `configure_nvidia_persistence.sh`,
`install_nvidia_container_toolkit.sh`) still run as bash, invoked by the Python
step exactly as the bash step did; `install_cuda_samples` runs as its Python
port, called by both driver steps.

`install-nv-driver` covers GB200/GB300 (and VR200, built as GB200) too, as in
bash: upstream #566 removed the dedicated `install_nvidiagpudriver_gb200.sh`, and
the shared script switches to IMEX/CDMM when `NVLINK_RACKSCALE=true` (packer sets
it for GB*/VR* SKUs).

### Known gap: distros other than Ubuntu 24.04

`build_plan()` follows `distros/ubuntu24.04/install.sh`. The other distros'
`install.sh` scripts differ from it in which component scripts they run — for
example Azure Linux uses `install_mofed.sh` instead of `install_doca.sh` and
doesn't run `install_lustre_client.sh`; AlmaLinux and Rocky don't run
`add-udev-rules.sh` and run extra scripts such as `network-config.sh`.
`DISTRO_DIRS` also names `almalinux9.7`/`rocky9.7` directories that don't exist
(they are `9.x`), and upstream added RHEL 8.10/9.x and Ubuntu 26.04, which the
CLI doesn't offer yet. Only the Ubuntu-only and Azure Linux cloud-init skips
are handled so far; the rest needs per-distro plans (ADO 991–997) before other
distros use the Python path.

---

## 13. Where this is headed

Beyond the component ports, three things carried over from the project's
original next steps:

1. **Prove it on an RPM distro.** The end-to-end build was Ubuntu
   (`Standard_NC24ads_A100_v4`, Ubuntu 24.04, zero errors, no pipeline changes).
   The `tdnf`/`dnf` path is tested but has never driven a real build.
2. **Wire the parity tests into CI**, so the bash and Python implementations
   can't silently drift.
3. **Port where Python adds value** — component by component, one reversible
   line at a time.

See `ToDo.md` in the repo root for the current open project-level items.

---

## 14. Reference

- Working branch: `modern/python`
- Build plan: `utils/build_config.py` → `build_plan()`
- Ported components: `components/python/`
- Tests: `tests/python/` — `python -m unittest discover -s tests/python`
- Entry point: `python3 azhpc.py --vendor NVidia --gpu A100 --os Ubuntu24`
