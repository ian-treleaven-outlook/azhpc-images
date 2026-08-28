# ToDo — before merging this fork upstream

Open items on the Python CLI port (`modern/python`). Items 1 and 2 were
deliberately deferred during the rebase onto `Azure/azhpc-images@f58ab80`;
item 3 was found while writing the contributor guide.

---

## 1. `install_rocm.sh` ordering diverges from `install.sh` (AMD path)

**What's wrong**

`build_plan()` in `utils/build_config.py` schedules ROCm inside the AMD branch,
*after* `install-dcgm`:

```python
# --- AMD branch ---
Step("install-rocm", "install_rocm.sh", when=amd),
Step("install-rccl", "install_rccl.sh", when=amd),
```

But `distros/ubuntu24.04/install.sh` runs it much earlier — before `install_pmix.sh`:

```
45: $COMPONENT_DIR/install_rocm.sh
49: $COMPONENT_DIR/install_pmix.sh
52: $COMPONENT_DIR/install_mpis.sh
```

So on AMD SKUs the Python plan installs ROCm *after* PMIx/MPI, while bash installs
it *before*. If PMIx or the MPI builds link against or detect ROCm, the Python path
could produce a materially different image.

**Why no test caught it**

`tests/python/test_build_plan.py` only asserts an ordered subsequence for the
**A100 (NVIDIA)** plan, which excludes `install_rocm.sh` entirely. The AMD plan is
checked for *membership* (`test_amd_branch`) but never for *order*.

**This predates the rebase** — it is not a regression from the upstream catch-up.
It was left alone to keep that change scoped to the actual regressions.

**Action**

- [ ] Move `install-rocm` ahead of `install-pmix` in `build_plan()`
- [ ] Add an ordered-subsequence parity test for the AMD/MI300 plan, mirroring
      `test_a100_plan_is_ordered_subsequence_of_install_sh`
- [ ] Re-check the other non-A100 plans (NCv6, GB200) for the same blind spot

---

## 2. CLA / commit authorship before opening the upstream PR

**Situation**

56 of the 57 commits on this branch are authored by:

```
Gio Martinez <giovannimart22@gmail.com>
```

That is a personal address belonging to an intern who has since left. The
authorship is correct and worth preserving for attribution — but it has
consequences for upstream submission.

**Why it matters**

- `Azure/azhpc-images` is a public Microsoft repo; its CLA/DCO check is evaluated
  against the **contributing identities on the commits**, not just the PR opener.
- Microsoft EMU accounts (`*_microsoft`) [cannot contribute to public open
  source](https://eng.ms/docs/initiatives/open-source-at-microsoft/github/microsoft-github),
  and [linking an EMU account
  fails](https://eng.ms/docs/initiatives/open-source-at-microsoft/github/opensource/accounts/linking)
  by design. The PR must come from a linked personal account —
  `ian-treleaven-outlook`, which is why this fork lives there.
- If Gio's account is inactive, he may be unable to sign the CLA after the fact,
  and maintainers cannot push to a branch owned by an unreachable account.

**Action**

- [ ] Confirm whether Gio's commits are already CLA-covered (work done as a
      Microsoft intern may be, via a corporate CLA — verify, don't assume)
- [ ] If not, decide the remedy *before* opening the PR: have him sign, or
      re-attribute with `Co-authored-by:` trailers preserving credit
- [ ] Confirm the PR is opened from `ian-treleaven-outlook`, not an EMU account

---

## 3. `install_cuda_samples` is ported but orphaned

**What's odd**

`components/python/install_cuda_samples.py` exists and exposes a complete
`install(env)`. Its bash counterpart `components/install_cuda_samples.sh` also
exists. But:

- **no distro's `install.sh` invokes it**, and
- there is **no corresponding `Step`** in `build_plan()`.

So it is a finished port of a component that is not part of any current build
flow. Unlike `components/python/install_nccl.py` — which documents itself as
`STAGED, NOT WIRED` with a clear reason — this one carries no such note, so the
intent is unclear.

**Action**

- [ ] Determine whether `install_cuda_samples.sh` is dead code upstream, or is
      meant to run on some path that currently misses it
- [ ] If it should run: add a `Step` with the correct `when=` gate and wire
      `action=install_cuda_samples.install`
- [ ] If it is genuinely dead: raise removal with upstream rather than deleting
      it locally — it is upstream's file, not ours

---

## Housekeeping

- [ ] **Delete this file before opening the upstream PR** — it is fork-local
      bookkeeping and shouldn't land in `Azure/azhpc-images`.

## Reference

- Working branch: `modern/python`
- Gio's pristine pre-rebase work (56 commits, tip `9ec3889`) is **no longer a
  branch on this fork** — it survives only in the mirror backup below. To read
  it back:
  `git --git-dir=D:\Gio\azhpc-images-backup.git log gio/python-cli`
- Full mirror backup of the original intern fork: `D:\Gio\azhpc-images-backup.git`
- Run the suite: `python -m unittest discover -s tests/python`
