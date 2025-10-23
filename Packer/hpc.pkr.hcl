# HPC Image Builder - Main Packer Template
# Author: ian-treleaven-outlook
# Date: 2025-10-23
# Description: Multi-target Packer template for HPC images with GPU support
#
# This Packer template demonstrates:
# - Multi-platform builds (Azure, Docker, Hyper-V, QEMU)
# - Variable usage and validation
# - Dynamic configuration with locals
# - Conditional provisioning
# - Post-processing steps

# ============================================================================
# VARIABLES SECTION
# Variables are input parameters that can be set at build time.
# They make your template reusable and configurable.
# Set via: -var "name=value" on command line or via .pkrvars.hcl files
# ============================================================================

# Build target determines which platform we're building for
# This affects which source block gets executed
variable "build_target" {
  type        = string
  description = "Build target platform (azure, docker, hyperv, qemu)"
  default     = "docker"
  
  # Validation ensures only valid values are accepted
  validation {
    condition     = contains(["azure", "docker", "hyperv", "qemu"], var.build_target)
    error_message = "Build target must be azure, docker, hyperv, or qemu."
  }
}

# OS configuration variables
# These determine what base image to use and how to configure it
variable "os_family" {
  type        = string
  description = "OS family (ubuntu, alma, azurelinux)"
  default     = "ubuntu"
  
  validation {
    condition     = contains(["ubuntu", "alma", "azurelinux"], var.os_family)
    error_message = "OS family must be ubuntu, alma, or azurelinux."
  }
}

variable "os_version" {
  type        = string
  description = "OS version (e.g., 22.04, 24.04, 8, 9)"
  default     = "22.04"
  
  # Version validation could be more specific per OS family
  # but keeping it flexible for now
}

variable "cpu_arch" {
  type        = string
  description = "CPU architecture (x86_64, arm64)"
  default     = "x86_64"
  
  validation {
    condition     = contains(["x86_64", "arm64"], var.cpu_arch)
    error_message = "CPU architecture must be x86_64 or arm64."
  }
}

variable "gpu_type" {
  type        = string
  description = "GPU type - determines which drivers to install"
  default     = "none"
  
  validation {
    condition = contains([
      "none",          # No GPU support
      "nvidia-a100",   # NVIDIA A100 (Azure NC A100 v4)
      "nvidia-h100",   # NVIDIA H100 (Azure NC H100 v5)
      "nvidia-gb200",  # NVIDIA Grace Blackwell
      "amd-mi300x"     # AMD Instinct MI300X
    ], var.gpu_type)
    error_message = "Invalid GPU type specified."
  }
}

# Docker-specific variable for custom base images
variable "docker_base_image" {
  type        = string
  description = "Override base Docker image (leave empty to use defaults)"
  default     = ""
}

# ============================================================================
# AZURE-SPECIFIC VARIABLES
# These variables are used when building for Azure.
# The env() function reads from environment variables if not explicitly set.
# This allows credentials to be passed securely without hardcoding.
# ============================================================================

variable "azure_client_id" {
  type        = string
  description = "Azure Service Principal Client ID (App ID)"
  default     = env("AZURE_CLIENT_ID")  # Reads from environment variable
}

variable "azure_client_secret" {
  type        = string
  description = "Azure Service Principal Client Secret (Password)"
  default     = env("AZURE_CLIENT_SECRET")
  sensitive   = true  # Marks as sensitive - won't be shown in logs
}

variable "azure_tenant_id" {
  type        = string
  description = "Azure AD Tenant ID"
  default     = env("AZURE_TENANT_ID")
}

variable "azure_subscription_id" {
  type        = string
  description = "Azure Subscription ID where resources will be created"
  default     = env("AZURE_SUBSCRIPTION_ID")
}

variable "azure_resource_group" {
  type        = string
  description = "Azure Resource Group for storing built images"
  default     = env("AZURE_RESOURCE_GROUP")
}

variable "azure_image_gallery" {
  type        = string
  description = "Azure Shared Image Gallery name for versioned images"
  default     = env("AZURE_IMAGE_GALLERY")
}

variable "location" {
  type        = string
  description = "Azure region for building and storing images"
  default     = "eastus"
}

# Optional: Temporary resource group for build VMs
variable "azure_build_resource_group" {
  type        = string
  description = "Temporary resource group for build resources (optional)"
  default     = ""  # If empty, uses azure_resource_group
}

# ============================================================================
# LOCAL VARIABLES SECTION
# Locals are computed values based on input variables.
# They help avoid repetition and create dynamic configurations.
# Think of them as "calculated fields" in your template.
# ============================================================================

locals {
  # Generate timestamp for unique naming
  # formatdate converts the timestamp to a readable format
  timestamp = formatdate("YYYY-MM-DD-hhmm", timestamp())
  
  # Map of OS combinations to Docker base images
  # This creates a lookup table for Docker base images
  docker_base_images = {
    "ubuntu-22.04" = "ubuntu:22.04"
    "ubuntu-24.04" = "ubuntu:24.04"
    "alma-8"       = "almalinux:8"
    "alma-9"       = "almalinux:9"
    "azurelinux-2" = "mcr.microsoft.com/azurelinux/base/core:2.0"
    "azurelinux-3" = "mcr.microsoft.com/azurelinux/base/core:3.0"
  }
  
  # Map of OS combinations to Azure marketplace images
  # Azure requires publisher/offer/sku to identify base images
  azure_base_images = {
    "ubuntu-22.04-x86_64" = {
      publisher = "Canonical"                    # Image publisher
      offer     = "0001-com-ubuntu-server-jammy" # Product offering
      sku       = "22_04-lts-gen2"              # Specific SKU (Gen2 = UEFI)
    }
    "ubuntu-24.04-x86_64" = {
      publisher = "Canonical"
      offer     = "ubuntu-24_04-lts"
      sku       = "server"
    }
    "alma-8-x86_64" = {
      publisher = "almalinux"
      offer     = "almalinux"
      sku       = "8-gen2"
    }
    "alma-9-x86_64" = {
      publisher = "almalinux"
      offer     = "almalinux"
      sku       = "9-gen2"
    }
    "azurelinux-2-x86_64" = {
      publisher = "MicrosoftCBLMariner"
      offer     = "cbl-mariner"
      sku       = "cbl-mariner-2-gen2"
    }
  }
  
  # Map GPU types to Azure VM sizes
  # Different GPUs require specific VM sizes in Azure
  azure_vm_sizes = {
    "x86_64-nvidia-a100" = "Standard_NC24ads_A100_v4"  # 1x A100 80GB
    "x86_64-nvidia-h100" = "Standard_NC40ads_H100_v5"  # 1x H100 80GB
    "x86_64-amd-mi300x"  = "Standard_NC40ahs_MI300X_v5" # 1x MI300X
    "x86_64-none"        = "Standard_D4s_v5"            # General purpose
    "arm64-none"         = "Standard_D4ps_v5"           # ARM64 general
  }
  
  # Build lookup keys for our maps
  base_image_key    = "${var.os_family}-${var.os_version}"
  azure_image_key   = "${var.os_family}-${var.os_version}-${var.cpu_arch}"
  azure_vm_size_key = "${var.cpu_arch}-${var.gpu_type}"
  
  # Lookup actual values from our maps with fallbacks
  # lookup() function: lookup(map, key, default)
  docker_base = lookup(local.docker_base_images, local.base_image_key, var.docker_base_image)
  azure_base  = lookup(local.azure_base_images, local.azure_image_key, {})
  azure_vm_size = lookup(local.azure_vm_sizes, local.azure_vm_size_key, "Standard_D4s_v5")
  
  # Generate consistent image naming
  # This creates a unique name for each build
  image_name = "hpc-${var.os_family}-${replace(var.os_version, ".", "-")}-${var.cpu_arch}-${var.gpu_type}-${local.timestamp}"
  
  # Shared Image Gallery version format (must be semantic versioning)
  # Azure requires versions like 1.0.0, so we use date-based versioning
  sig_version = formatdate("YYYY.MMDD.hhmm", timestamp())
}

# ============================================================================
# PACKER CONFIGURATION BLOCK
# This block defines Packer version requirements and required plugins.
# Plugins extend Packer's functionality for different platforms.
# ============================================================================

packer {
  # Minimum Packer version required
  # This ensures users have compatible features
  required_version = ">= 1.9.0"
  
  # Required plugins declaration
  # Packer will automatically download these if missing
  required_plugins {
    # Docker plugin for container image builds
    docker = {
      version = ">= 1.0.0"
      source  = "github.com/hashicorp/docker"
    }
    
    # Azure plugin for Azure VM image builds
    azure = {
      version = ">= 2.0.0"
      source  = "github.com/hashicorp/azure"
    }
    
    # Ansible plugin for configuration management (optional)
    ansible = {
      version = ">= 1.1.0"
      source  = "github.com/hashicorp/ansible"
    }
  }
}

# ============================================================================
# SOURCE BLOCKS
# Sources define WHERE and HOW to build images.
# Each source is a different platform/method for building.
# The source name format is: source "type" "name"
# Referenced later as: source.type.name
# ============================================================================

# Docker source - builds container images locally
source "docker" "base" {
  # Base image to start from
  image  = local.docker_base
  
  # Commit the container as an image (vs export as tar)
  commit = true
  
  # Metadata to add to the image
  # These become Docker labels
  changes = [
    "LABEL version=${local.timestamp}",
    "LABEL os=${var.os_family}-${var.os_version}",
    "LABEL arch=${var.cpu_arch}",
    "LABEL gpu=${var.gpu_type}",
    "LABEL maintainer=ian-treleaven-outlook",
    "LABEL build.date=${local.timestamp}",
    "LABEL build.tool=packer-${packer.version}"
  ]
  
  # How to run the container during build
  # --privileged needed for some system operations
  run_command = [
    "-d",                              # Detached mode
    "-t",                              # Allocate pseudo-TTY
    "--name", "packer-${local.image_name}", # Container name
    "--privileged",                    # Full privileges (needed for some installs)
    "{{ .Image }}"                     # Image placeholder - Packer fills this
  ]
}

# Azure source - builds VM images in Azure
source "azure-arm" "base" {
  # ===== Authentication =====
  # These credentials are used to connect to Azure
  # In ADO pipeline, these come from service connection
  client_id       = var.azure_client_id
  client_secret   = var.azure_client_secret
  tenant_id       = var.azure_tenant_id
  subscription_id = var.azure_subscription_id
  
  # ===== Source Image Configuration =====
  # Defines the base OS image from Azure Marketplace
  # try() function provides graceful fallback if key doesn't exist
  image_publisher = try(local.azure_base.publisher, "Canonical")
  image_offer     = try(local.azure_base.offer, "0001-com-ubuntu-server-jammy")
  image_sku       = try(local.azure_base.sku, "22_04-lts-gen2")
  image_version   = "latest"  # Always use latest patch version
  
  # ===== Build VM Configuration =====
  # Temporary VM used for building the image
  location = var.location
  vm_size  = local.azure_vm_size  # Size based on GPU requirements
  
  # Optional: Use separate resource group for build resources
  # This keeps build artifacts separate from final images
  build_resource_group_name = var.azure_build_resource_group != "" ? var.azure_build_resource_group : var.azure_resource_group
  
  # ===== Managed Image Output =====
  # Creates a managed image in the specified resource group
  managed_image_resource_group_name = var.azure_resource_group
  managed_image_name                 = local.image_name
  
  # ===== Shared Image Gallery Output =====
  # Also publishes to SIG for versioning and replication
  shared_image_gallery_destination {
    subscription   = var.azure_subscription_id
    resource_group = var.azure_resource_group
    gallery_name   = var.azure_image_gallery
    
    # Image definition name (like a "product" in the gallery)
    image_name     = "hpc-${var.os_family}-${replace(var.os_version, ".", "-")}-${var.cpu_arch}-${var.gpu_type}"
    
    # Version of this image (must be semantic version format)
    image_version  = local.sig_version
    
    # Replication settings (optional)
    # target_regions = ["eastus", "westus2"]  # Replicate to multiple regions
  }
  
  # ===== OS Disk Configuration =====
  os_type         = "Linux"
  os_disk_size_gb = 128  # Size in GB for OS disk
  
  # ===== Network Configuration =====
  # Uses default Azure virtual network
  # Could specify custom VNet if needed:
  # virtual_network_name = "my-vnet"
  # virtual_network_subnet_name = "my-subnet"
  
  # ===== Resource Tags =====
  # Tags help with organization and cost tracking
  azure_tags = {
    OS          = "${var.os_family}-${var.os_version}"
    GPU         = var.gpu_type
    Architecture = var.cpu_arch
    Builder     = "packer"
    BuildDate   = local.timestamp
    Owner       = "ian-treleaven-outlook"
    Team        = "hpc-platform-team"
    Purpose     = "HPC-Image-Testing"
    Source      = "packer-template"
  }
  
  # ===== SSH Configuration =====
  # Packer uses SSH to connect and run provisioners
  # Uses default generated credentials
  # Could override with:
  # ssh_username = "packer"
  # ssh_password = "ComplexP@ssw0rd!"
}

# ============================================================================
# BUILD BLOCKS
# Build blocks tie together sources and provisioners.
# They define WHAT gets built and HOW it's configured.
# You can have multiple build blocks for different image types.
# ============================================================================

build {
  # Build name - useful for filtering with -only flag
  name = "base"
  
  # Description for documentation
  description = "HPC base image with GPU support"
  
  # ===== Sources =====
  # List which sources this build applies to
  # Format: "source.type.name"
  sources = [
    "source.docker.base",
    "source.azure-arm.base"
  ]
  
  # ===== Provisioners =====
  # Provisioners run IN ORDER to configure the image
  # Each provisioner is a step in the build process
  
  # Step 1: Base OS setup
  # This script updates packages and installs basic tools
  provisioner "shell" {
    script = "scripts/setup-base.sh"
    
    # Environment variables passed to the script
    # Scripts can access these as $VAR_NAME
    environment_vars = [
      "OS_FAMILY=${var.os_family}",
      "OS_VERSION=${var.os_version}",
      "CPU_ARCH=${var.cpu_arch}"
    ]
    
    # Optional: Control when this runs
    # only   = ["docker.base"]  # Only run for Docker
    # except = ["azure-arm.base"]  # Skip for Azure
    
    # Optional: Pause before running (debugging)
    # pause_before = "10s"
    
    # Optional: Expect certain exit codes
    # valid_exit_codes = [0, 1]
  }
  
  # Step 2: HPC common tools
  # Installs MPI, development tools, etc.
  provisioner "shell" {
    script = "scripts/hpc-common.sh"
    environment_vars = [
      "OS_FAMILY=${var.os_family}",
      "OS_VERSION=${var.os_version}"
    ]
    
    # Optional: Run with elevated privileges
    # execute_command = "chmod +x {{ .Path }}; sudo '{{ .Path }}'"
  }
  
  # Step 3: GPU drivers (conditional)
  # Only runs if GPU type is not "none"
  # Demonstrates conditional provisioning
  provisioner "shell" {
    script = "scripts/install-gpu-drivers.sh"
    environment_vars = [
      "GPU_TYPE=${var.gpu_type}",
      "OS_FAMILY=${var.os_family}",
      "OS_VERSION=${var.os_version}"
    ]
    
    # Conditional execution based on GPU type
    # This is a complex conditional using ternary operator
    # Format: condition ? [run-on-these] : [skip-these]
    only = var.gpu_type != "none" ? ["docker.base", "azure-arm.base"] : []
  }
  
  # Step 4: Infiniband/RDMA setup
  # For high-speed networking in HPC clusters
  provisioner "shell" {
    script = "scripts/setup-infiniband.sh"
    environment_vars = [
      "OS_FAMILY=${var.os_family}",
      "OS_VERSION=${var.os_version}"
    ]
    
    # Optional: Skip on failure (best effort)
    # on_error = "continue"
  }
  
  # Alternative: Inline commands
  # Good for simple operations
  provisioner "shell" {
    inline = [
      "echo 'Running inline commands'",
      "df -h",  # Check disk space
      "free -m", # Check memory
      "lscpu"   # Check CPU info
    ]
  }
  
  # Step 5: Final cleanup
  # Reduces image size by removing temporary files
  provisioner "shell" {
    inline = [
      "echo 'Cleaning up temporary files...'",
      
      # Clean package manager cache
      "if command -v apt-get >/dev/null; then apt-get clean && rm -rf /var/lib/apt/lists/*; fi",
      "if command -v yum >/dev/null; then yum clean all; fi",
      
      # Clean temporary files
      "rm -rf /tmp/* /var/tmp/*",
      
      # Clean bash history
      "rm -f /root/.bash_history",
      
      # Zero out free space (reduces image size for some platforms)
      # "dd if=/dev/zero of=/EMPTY bs=1M || true; rm -f /EMPTY",
      
      # Sync filesystem
      "sync"
    ]
  }
  
  # ===== Post-Processors =====
  # Post-processors run AFTER provisioning completes
  # They handle the final image artifact
  
  # Docker: Tag the image with multiple tags
  post-processor "docker-tag" {
    repository = "hpc-images/${var.os_family}"
    
    # Multiple tags for the same image
    tags = [
      "${var.os_version}-${var.gpu_type}-latest",           # Latest tag
      "${var.os_version}-${var.gpu_type}-${local.timestamp}" # Timestamped tag
    ]
    
    # Only run for Docker builds
    only = ["docker.base"]
  }
  
  # Optional: Push to Docker registry
  # post-processor "docker-push" {
  #   login          = true
  #   login_username = var.docker_username
  #   login_password = var.docker_password
  #   only          = ["docker.base"]
  # }
  
  # Optional: Create manifest file
  post-processor "manifest" {
    output = "manifest.json"
    
    # Custom data to include in manifest
    custom_data = {
      build_time = local.timestamp
      os_info    = "${var.os_family}-${var.os_version}"
      gpu_type   = var.gpu_type
      builder    = "ian-treleaven-outlook"
    }
  }
  
  # Optional: Compress artifact
  # post-processor "compress" {
  #   output = "{{.BuildName}}-{{.Provider}}.tar.gz"
  # }
}

# ============================================================================
# ADDITIONAL BUILD BLOCKS (Optional)
# You can have multiple build blocks for different configurations
# Example: Separate build for development vs production
# ============================================================================

# build {
#   name = "development"
#   
#   sources = ["source.docker.base"]
#   
#   # Development-specific provisioning
#   provisioner "shell" {
#     inline = [
#       "echo 'Installing development tools...'",
#       "apt-get install -y gdb valgrind strace"
#     ]
#   }
# }

# ============================================================================
# ADVANCED FEATURES (Comments for Learning)
# ============================================================================

# 1. HCL2 Functions Available:
#    - abs, ceil, floor, log, max, min, pow, signum
#    - chomp, format, formatlist, indent, join, lower, regex, split, trim, upper
#    - chunklist, coalesce, coalescelist, compact, concat, contains, distinct, element
#    - file, fileset, pathexpand, basename, dirname
#    - jsondecode, jsonencode, yamldecode, yamlencode
#    - timestamp, formatdate

# 2. Conditional Logic:
#    - Use ternary operators: condition ? true_value : false_value
#    - Use only/except in provisioners for conditional execution

# 3. Dynamic Blocks:
#    - Can generate multiple similar blocks programmatically
#    - Useful for multiple regions, multiple images, etc.

# 4. Data Sources:
#    - Can query external data during build
#    - Example: Query latest AMI ID from AWS

# 5. Build Variables:
#    - build.name: Current build name
#    - build.type: Builder type (docker, azure-arm, etc.)
#    - source.name, source.type: Source information
#    - packer.version: Packer version

# 6. Error Handling:
#    - on_error = "abort" (default) | "continue" | "cleanup"
#    - max_retries and retry_delay for network operations

# 7. Debugging:
#    - Set PACKER_LOG=1 environment variable
#    - Use -debug flag for step-by-step execution
#    - Add pause_before/pause_after to provisioners