"""Python counterparts to the components/install_*.sh scripts.

The entry point is install(env) -> int, returning 0 on success and 3 on failure;
utils/plans/steps.py hangs those off a Step as `action=`, and each distro's
plan (utils/plans/) lists the steps. install_cuda_samples has no Step of its own: as in bash, the two
driver steps (install_nvidiagpudriver, install_nvidiagriddriver) call it.
"""
