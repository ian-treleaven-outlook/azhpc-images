# HPC Image Builder

**Author**: ian-treleaven-outlook  
**Date**: 2025-10-23  
**Purpose**: Build optimized HPC images for Azure Marketplace

## Overview

This project provides a unified Packer-based solution for building HPC-optimized VM images with GPU support across multiple platforms.

## Features

- **Multi-platform support**: Azure, Docker, Hyper-V, QEMU
- **Multiple OS families**: Ubuntu, AlmaLinux, Azure Linux
- **GPU support**: NVIDIA (A100, H100), AMD (MI300X)
- **HPC optimizations**: MPI, RDMA, Infiniband
- **CI/CD integration**: Azure DevOps pipelines

## Quick Start

### Local Development (Docker)

```bash
# Build Ubuntu 22.04 with NVIDIA A100 support
./build.sh -t docker -o ubuntu -v 22.04 -g nvidia-a100

# Build AlmaLinux 9 base image
./build.sh -t docker -o alma -v 9