"""Python counterparts to the components/install_*.sh scripts.

The entry point is install(env) -> int, returning 0 on success and 3 on failure;
build_config.py hangs those off a Step as `action=`. Modules with a package
dependency phase also have install_deps(env), but that's an internal helper that
install() calls itself, not something the build plan invokes.

Not every module here is wired up yet -- install_nccl and install_cuda_samples
are staged; see the note at the top of each.
"""
