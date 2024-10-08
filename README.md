# English Guide
## Features
Dependency Checks: Verifies the installation and minimum required versions of essential packages.
System Information: Displays the current operating system and Python version.
Hardware Checks: Confirms the availability of CUDA, cuDNN, and GPUs.
Multiprocessing Support: Checks if multiprocessing is available.
Distributed Backend Availability: Determines if the appropriate backend (NCCL or Gloo) is available for distributed training.
Distributed Training Test: Performs a simple distributed tensor operation to ensure that distributed training is functional.
Custom Dependency File Support: Allows you to specify a requirements.txt file to check additional dependencies.
