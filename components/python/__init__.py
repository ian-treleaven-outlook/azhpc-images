"""Python counterparts to the components/install_*.sh scripts.

The entry point is install(env) -> int, returning 0 on success and 3 on failure;
build_config.py hangs those off a Step as `action=`. Every step in build_plan()
now has one. install_cuda_samples has no Step of its own: as in bash, the two
driver steps (install_nvidiagpudriver, install_nvidiagriddriver) call it.
"""
