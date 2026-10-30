from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from utils.logger import log_info, log_warn, log_error, log_error_detail
from utils.process import exec_program
import os
import platform
import subprocess
import time
import traceback

from utils.plans import OS_TARGETS, PENDING, PLANS
from utils.plans.step import Step
from utils.sku import sku_family
from utils.system_prep import prepare_system
from utils.component_config import DEFAULT_NODE_TYPE

MODULE_DIRS = {
    "ubuntu": "/usr/share/modules/modulefiles",
}
DEFAULT_MODULE_DIR = "/usr/share/Modules/modulefiles"  # alma/rocky/azure/rhel

VALID_GPUS = {
    "NVidia": {"GB200", "GB300", "NCv6", "V100", "A100", "H100", "H200", "VR200"},
    # MI300X is the SKU Packer passes for Standard_ND96isr_MI300X_v5 (bash then
    # exports SKU=MI300X); MI300 is kept for existing command lines.
    "AMD": {"MI300X", "MI300", "MI400", "MI500"},
}
DEDICATED_PATH_GPUS = {"GB200", "GB300", "NCv6", "V100"}


def is_rackscale(gpu: str) -> bool:
    """Grace/Vera rackscale NVLink SKUs (GB200, GB300, VR200): aarch64, and
    NVLINK_RACKSCALE=true. The same test Packer uses (startswith GB / VR)."""
    return gpu.startswith(("GB", "VR"))


def build_sku(gpu: str, os_name: str | None = None) -> str:
    """The SKU bash actually builds for `gpu` on `os_name`.

    Some install.sh scripts build one SKU as another: Ubuntu 24.04's turns
    VR200 into GB200 (`export SKU=$([[ $2 == "VR200" ]] && echo "GB200" ...)`).
    Each distro plan lists its own (DistroPlan.sku_aliases); any other SKU,
    or an OS without a plan, is returned unchanged.
    """
    plan = PLANS.get(os_name) if os_name else None
    return plan.sku_aliases.get(gpu, gpu) if plan else gpu

# --os value -> its distros/ folder, and the distribution versions.json uses
# for it (see TargetOS in utils/plans/__init__.py).
DISTRO_DIRS = {name: f"distros/{target.folder}" for name, target in OS_TARGETS.items()}
DISTRIBUTIONS = {name: target.distribution for name, target in OS_TARGETS.items()}
GPU_ARGS = {"NVidia": "NVIDIA", "AMD": "AMD"}

def _detect_distribution() -> str:
    """Mirror: . /etc/os-release; echo $ID$VERSION_ID  -> e.g. 'ubuntu24.04'."""
    data: dict[str, str] = {}
    with open("/etc/os-release", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                data[key] = val.strip().strip('"')
    return f"{data.get('ID', '')}{data.get('VERSION_ID', '')}"

def _detect_arch_distro(distribution: str) -> str:
    """Mirror: dpkg --print-architecture (deb) or rpm --eval %{_arch} (rpm)."""
    if "ubuntu" in distribution:
        cmd = ["dpkg", "--print-architecture"]
    else:
        cmd = ["rpm", "--eval", "%{_arch}"]
    return subprocess.run(cmd, capture_output
                          =True, text=True).stdout.strip()

def component_env(repo_root: Path, cfg: BuildConfig) -> dict[str, str]:
    """The env vars set_properties.sh exported that the component scripts read.

    Pure environment setup only; set_properties.sh's system changes (apt
    upgrade, jq, ...) run in the bootstrap (utils/system_prep.py).
    """
    distribution = _detect_distribution()

    env = os.environ.copy()
    env["TOP_DIR"]       = str(repo_root)
    env["COMPONENT_DIR"] = str(repo_root / "components")
    env["UTILS_DIR"]     = str(repo_root / "utils")
    env["TEST_DIR"]      = str(repo_root / "tests")
    # The same directory under the name set_properties.sh exports and
    # copy_test_file.sh / copy_test_file.py read. Without it the bash script
    # would copy from "/" instead of the repo's tests (the Python one fails).
    env["AZHPC_IMAGES_TEST_DIR"] = str(repo_root / "tests")

    env["DISTRIBUTION"]        = distribution
    env["ARCHITECTURE"]        = platform.machine()              # uname -m
    env["ARCHITECTURE_DISTRO"] = _detect_arch_distro(distribution)

    env["GPU"]        = cfg.gpu_arg                              # NVIDIA / AMD
    env["SKU"]        = cfg.gpu                                  # H100, MI300X, ...
    # As set_properties.sh: keep a SKU_FAMILY the caller already exported
    # (`if [[ -z "${SKU_FAMILY:-}" ]]`), otherwise derive it from the SKU.
    env["SKU_FAMILY"] = os.environ.get("SKU_FAMILY") or sku_family(cfg.gpu)
    # The deployment target, under the same name and default bash uses (see
    # utils/set_properties.sh). versions.json overrides are keyed on it, and the
    # bash steps that still run read it directly. An empty value gets the
    # default, matching bash's ${TARGET_NODE_TYPE:-azure_vm_regular}.
    env["TARGET_NODE_TYPE"] = os.environ.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE

    env["MODULE_FILES_DIRECTORY"] = (
        MODULE_DIRS["ubuntu"] if "ubuntu" in distribution else DEFAULT_MODULE_DIR
    )

    # Keep apt/dpkg out of interactive mode. Without this, debconf takes over the
    # terminal to draw prompts and progress, which corrupts the log's formatting.
    env["DEBIAN_FRONTEND"] = "noninteractive"

    versions = repo_root / "versions.json"
    if versions.is_file():
        env["COMPONENT_VERSIONS"] = versions.read_text(encoding="utf-8")

    return env

@dataclass(frozen=True)
class BuildConfig:
    vendor: str
    gpu: str
    os: str
    fips: bool
    spec_path: Path | None

    @property
    def distro_dir(self) -> str:
        return DISTRO_DIRS[self.os]
    
    @property
    def gpu_arg(self) -> str:
        return GPU_ARGS[self.vendor]
    
    @property
    def has_dedicated_path(self) -> bool:
        return self.gpu in DEDICATED_PATH_GPUS

    @property
    def architecture(self) -> str:
        """Target CPU architecture, derived from the SKU (Grace/Vera SKUs are ARM)."""
        return "aarch64" if is_rackscale(self.gpu) else "x86_64"

class ConfigError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code

def resolve_config(args) -> BuildConfig:
    # required args
    if not args.vendor:
        raise ConfigError("--vendor is required", 1)
    if not args.gpu:
        raise ConfigError("--gpu is required", 1)
    if not args.os:
        raise ConfigError("--os is required", 1)

    # vendor / gpu / os validation 
    if args.vendor not in VALID_GPUS:
        raise ConfigError(f"unsupported vendor '{args.vendor}' (NVidia|AMD)", 1)
    if args.gpu not in VALID_GPUS[args.vendor]:
        raise ConfigError(f"GPU '{args.gpu}' not valid for vendor {args.vendor}", 1)
    if args.os not in DISTRO_DIRS:
        raise ConfigError(
            f"unsupported os '{args.os}' ({'|'.join(DISTRO_DIRS)})", 1)
    # Every --os has a distro folder, but not every one has a Python build
    # plan yet. Building one without a plan would run the wrong install.sh.
    if args.os not in PLANS:
        raise ConfigError(
            f"no Python build plan for {args.os} ({DISTRO_DIRS[args.os]}/install.sh) "
            f"yet ({PENDING.get(args.os, 'not planned')}); build it with bash "
            f"(Python plans: {', '.join(sorted(PLANS))})", 1)
    plan = PLANS[args.os]
    if args.vendor not in plan.vendors:
        raise ConfigError(
            plan.vendor_error or f"{args.vendor} is not supported on {args.os}", 1)

    # spec file validation
    spec_path: Path | None = None
    if args.spec:
        spec_path = Path(args.spec)
        if not spec_path.is_file():
            raise ConfigError(f"spec file not found: {args.spec}", 2)
        try:
            json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raise ConfigError(f"spec file is malformed JSON: {args.spec}", 2)

    # Placeholder SKUs some install.sh scripts build as another SKU
    # (DistroPlan.sku_aliases). The requested name stays in every log line's
    # "gpu" field (azhpc.py sets it before this runs), so telemetry still shows
    # VR200 was asked for.
    gpu = build_sku(args.gpu, args.os)
    if gpu != args.gpu:
        log_warn("resolve-config",
                 f"GPU '{args.gpu}' is built as {gpu}: a placeholder mapping copied "
                 f"from {DISTRO_DIRS[args.os]}/install.sh",
                 requested_gpu=args.gpu, build_gpu=gpu)
    # SKUs this distro's install.sh refuses, before it does anything.
    error = plan.reject_sku(gpu)
    if error:
        raise ConfigError(error, 1)

    # build and return the immutable config object
    return BuildConfig(
        vendor=args.vendor,
        gpu=gpu,
        os=args.os,
        fips=bool(args.fips),
        spec_path=spec_path,
    )

def build_plan(cfg: BuildConfig) -> list[Step]:
    """The ordered steps after install_utils.sh, mirroring the distro's install.sh.

    Each distro has its own plan in utils/plans/ (e.g. ubuntu24_04.py for
    distros/ubuntu24.04/install.sh), built from the shared step library in
    utils/plans/steps.py. Every step carries its canonical bash script name
    (for parity), plus the Python `action` that runs it natively.
    """
    return _plan_for(cfg).build(cfg)


def pre_bootstrap_plan(cfg: BuildConfig) -> list[Step]:
    """The steps the distro's install.sh runs before install_utils.sh (usually none)."""
    return _plan_for(cfg).pre_bootstrap(cfg)


def _plan_for(cfg: BuildConfig):
    if cfg.os not in PLANS:
        # resolve_config() rejects these first; this guards direct callers.
        raise ConfigError(f"no Python build plan for {cfg.os} yet "
                          f"({PENDING.get(cfg.os, 'not planned')})", 1)
    return PLANS[cfg.os]

def _new_counts() -> dict[str, int]:
    return {"python": 0, "bash": 0, "skipped": 0}


class ImageBuilder:
    def __init__(self, repo_root: Path, config: BuildConfig):
        self.repo_root = repo_root
        self.config = config

    def _bootstrap(self, build_dir: Path, env: dict[str, str],
                   counts: dict[str, int] | None = None) -> int:
        """Run system prep that must happen before any component.

        As every distro's install.sh does: first the system changes
        utils/set_properties.sh makes (apt update and upgrade, jq, AlmaLinux
        GPG keys; see utils/system_prep.py), then any steps the distro's
        install.sh runs before install_utils.sh (e.g. Ubuntu 22.04's
        remove_unused_packages.sh), then install_utils.sh itself (Microsoft
        repo, base packages, IB modules). install_utils.sh is still bash: it's
        heavily distro-specific, side-effecting work.
        """
        rc = prepare_system(env)
        if rc != 0:
            return rc
        pre = pre_bootstrap_plan(self.config)
        if pre:
            rc = self._run_steps(pre, env, build_dir,
                                 counts if counts is not None else _new_counts())
            if rc != 0:
                return rc
        install_utils = build_dir / "install_utils.sh"
        log_info("bootstrap", "Preparing system (install_utils.sh)")
        return exec_program([str(install_utils)], "bootstrap",
                            cwd=str(build_dir), env=env)

    def _run_steps(self, steps: list[Step], env: dict[str, str], build_dir: Path,
                   counts: dict[str, int]) -> int:
        """Run `steps` in order. Returns 0, or 3 at the first failure.

        `counts` (python / bash / skipped) is updated for the summary line.
        """
        cfg = self.config
        for step in steps:
            # (a) skip steps that don't apply to this target (e.g. rocm on NVIDIA).
            # A skip that may hide a problem (e.g. install.sh for this distro
            # never runs the script) is logged as a WARNING with the reason.
            if not step.when(cfg):
                counts["skipped"] += 1
                if step.skip_warning:
                    log_warn(step.op, f"skipped: {step.skip_warning}",
                             os=cfg.os, script=step.script)
                else:
                    log_info(step.op, "skipped: not applicable to this target")
                continue

            # Record whether this step ran as Python or bash, so a failure in a
            # real build can be traced to the right implementation. A port
            # marked `resync` is out of date, so its bash script runs instead.
            runner = step.runner
            if step.action is not None and step.resync:
                log_warn(step.op,
                         f"Python port is behind {step.script} ({step.resync}); "
                         "running the bash script instead",
                         script=step.script, resync=step.resync)
            log_info(step.op, f"starting ({runner})", runner=runner, script=step.script)
            step_started = time.monotonic()

            # (b) run it: a python action, or the bash script via exec_program.
            # A python action that raises (bad tarball, missing file, ...) is
            # turned into a normal rc=3 failure here, so an unexpected exception
            # can't escape as a traceback and break the documented exit codes.
            # The traceback is still logged, as the error's detail block.
            try:
                if runner == "python":
                    rc = step.action(env)
                else:
                    # The script is in components/, utils/ or the distro folder
                    # (Step.base); like install.sh, it runs in the distro folder.
                    script = step.script_path(self.repo_root, cfg.distro_dir)
                    rc = exec_program([str(script), *step.args],
                                      step.op, cwd=str(build_dir), env=env)
            except Exception as exc:
                elapsed = time.monotonic() - step_started
                log_error_detail(
                    step.op,
                    f"unhandled error after {elapsed:.1f}s: "
                    f"{exc.__class__.__name__}: {exc}",
                    traceback.format_exc(),
                    runner=runner, duration_s=round(elapsed, 1))
                return 3

            # (c) stop the whole build on the first failure
            elapsed = time.monotonic() - step_started
            if rc != 0:
                log_error(step.op, f"failed with exit code {rc} after {elapsed:.1f}s",
                          runner=runner, exit_code=rc, duration_s=round(elapsed, 1))
                return 3
            counts[runner] += 1
            log_info(step.op, f"done in {elapsed:.1f}s",
                     runner=runner, duration_s=round(elapsed, 1))
        return 0

    def build(self) -> int:
        cfg = self.config

        # Warn (but continue) if this GPU uses the generic path rather than a
        # dedicated one (GB200/NCv6/... have special handling; A100 does not).
        if not cfg.has_dedicated_path:
            log_warn("resolve-config",
                     f"GPU '{cfg.gpu}' has no dedicated build path; "
                     f"using generic {cfg.vendor} build")

        # --- Phase 1: setup ---
        # build_dir  = the distro folder the bash scripts run from
        # env        = vars handed to every step (this loads versions.json)
        build_dir = self.repo_root / cfg.distro_dir
        env = component_env(self.repo_root, cfg)

        plan = build_plan(cfg)
        build_started = time.monotonic()
        # Fields on this line describe the whole run, so a log query can find
        # every build of, say, NCv6 on azurelinux3.0.
        log_info("build-image", f"Building {cfg.os} for {cfg.gpu}",
                 vendor=cfg.vendor, gpu=cfg.gpu, os=cfg.os,
                 distribution=env.get("DISTRIBUTION", ""),
                 architecture=env.get("ARCHITECTURE", ""),
                 node_type=env.get("TARGET_NODE_TYPE", ""),
                 planned_steps=len(plan),
                 pre_bootstrap_steps=len(pre_bootstrap_plan(cfg)))

        # Rackscale NVLink SKUs rely on NVLINK_RACKSCALE=true, which packer sets
        # (startswith "GB"/"VR") and the shared driver/tuning scripts read. Like
        # bash, we don't derive it ourselves; but a manual build without it
        # silently gets the non-rackscale driver path, so say so.
        if is_rackscale(cfg.gpu) and env.get("NVLINK_RACKSCALE", "").lower() != "true":
            log_warn("resolve-config",
                     f"NVLINK_RACKSCALE is not 'true' for {cfg.gpu}: the driver step "
                     "will install fabric manager instead of IMEX/CDMM, as bash would "
                     "(packer normally sets it for GB*/VR* SKUs)",
                     gpu=cfg.gpu, nvlink_rackscale=env.get("NVLINK_RACKSCALE", ""))

        # --- Phase 2: bootstrap (system prep) ---
        # set_properties.sh's changes, any pre-bootstrap steps, then
        # install_utils.sh; abort the build if it fails.
        counts = _new_counts()          # for the summary line at the end
        bootstrap_started = time.monotonic()
        rc = self._bootstrap(build_dir, env, counts)
        elapsed = time.monotonic() - bootstrap_started
        if rc != 0:
            log_error("bootstrap", f"system prep failed with exit code {rc} after {elapsed:.1f}s",
                      exit_code=rc, duration_s=round(elapsed, 1))
            return 3
        log_info("bootstrap", f"done in {elapsed:.1f}s", duration_s=round(elapsed, 1))

        # --- Phase 3: run each step of the plan, in order ---
        if self._run_steps(plan, env, build_dir, counts) != 0:
            return 3

        # --- Phase 4: every step passed ---
        total = time.monotonic() - build_started
        log_info("build-image", f"Image built successfully in {total:.1f}s",
                 duration_s=round(total, 1), python_steps=counts["python"],
                 bash_steps=counts["bash"], skipped_steps=counts["skipped"])
        return 0

        