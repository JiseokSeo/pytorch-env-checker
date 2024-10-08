import sys
import platform
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

# Import importlib.metadata or importlib_metadata if necessary
try:
    from importlib import metadata
except ImportError:
    # For Python versions below 3.8
    import importlib_metadata as metadata

from packaging import version  # Needed for version comparison


def parse_args():
    """Parse command-line arguments to set MASTER_ADDR, MASTER_PORT, and dependencies."""
    parser = argparse.ArgumentParser(description="PyTorch Environment Check")
    parser.add_argument(
        "--ADDR",
        type=str,
        default="127.0.0.1",
        help="Set MASTER_ADDR environment variable",
    )
    parser.add_argument(
        "--PORT",
        type=str,
        default="29500",
        help="Set MASTER_PORT environment variable",
    )
    parser.add_argument(
        "--dependencies",
        type=str,
        default=None,
        help="Specify a requirements.txt file to check dependencies",
    )
    args = parser.parse_args()
    return args.ADDR, args.PORT, args.dependencies


def check_os():
    """Check and return the current operating system."""
    os_name = platform.system()
    message = f"{os_name}"
    return os_name, message


def check_python_version():
    """Check the Python version."""
    version_info = sys.version
    message = f"Python Version: {version_info}"
    return True, message


def check_dependencies(requirements_file):
    """Check dependencies from the requirements.txt file."""
    if not requirements_file:
        message = "No dependencies file specified. Skipping dependencies check."
        return True, message

    if not os.path.exists(requirements_file):
        message = f"Dependencies file '{requirements_file}' not found."
        return False, message

    missing_packages = []
    incorrect_versions = []

    with open(requirements_file, "r") as f:
        for line in f:
            # Ignore comments and empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse package and version
            comparator = None
            if "==" in line:
                package, required_version = line.split("==")
                comparator = "=="
            elif ">=" in line:
                package, required_version = line.split(">=")
                comparator = ">="
            elif ">" in line:
                package, required_version = line.split(">")
                comparator = ">"
            elif "<=" in line:
                package, required_version = line.split("<=")
                comparator = "<="
            elif "<" in line:
                package, required_version = line.split("<")
                comparator = "<"
            else:
                package = line
                required_version = None

            package = package.strip()
            if required_version:
                required_version = required_version.strip()

            try:
                installed_version = metadata.version(package)
                if required_version and comparator:
                    if not compare_versions(
                        installed_version, required_version, comparator
                    ):
                        incorrect_versions.append(
                            (package, installed_version, comparator, required_version)
                        )
            except metadata.PackageNotFoundError:
                missing_packages.append(package)

    if missing_packages or incorrect_versions:
        message = ""
        if missing_packages:
            message += f"Missing packages: {', '.join(missing_packages)}. "
        if incorrect_versions:
            for pkg, installed, comp, required in incorrect_versions:
                message += f"{pkg} version {installed} does not satisfy the requirement {comp} {required}. "
        message += "Please install the required packages."
        return False, message
    else:
        message = "All required dependencies are installed with correct versions."
        return True, message


def compare_versions(installed_version, required_version, comparator):
    """Compare versions to check if the requirement is satisfied."""
    installed = version.parse(installed_version)
    required = version.parse(required_version)

    if comparator == "==":
        return installed == required
    elif comparator == ">=":
        return installed >= required
    elif comparator == ">":
        return installed > required
    elif comparator == "<=":
        return installed <= required
    elif comparator == "<":
        return installed < required
    else:
        return False  # Unknown comparator


def check_pytorch():
    """Check if PyTorch is installed and get its version."""
    try:
        version_info = torch.__version__
        message = f"PyTorch is installed. Version: {version_info}"
        return True, message
    except ImportError as e:
        message = f"PyTorch is not installed: {e}"
        return False, message


def check_cuda():
    """Check if CUDA is available and get its version."""
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            message = f"CUDA is available. Version: {cuda_version}"
            return True, message
        else:
            message = "CUDA is not available."
            return False, message
    except Exception as e:
        message = f"Error checking CUDA availability: {e}"
        return False, message


def check_cudnn():
    """Check if cuDNN is available and get its version."""
    try:
        cudnn_available = torch.backends.cudnn.is_available()
        if cudnn_available:
            cudnn_version = torch.backends.cudnn.version()
            message = f"cuDNN is available. Version: {cudnn_version}"
            return True, message
        else:
            message = "cuDNN is not available."
            return False, message
    except Exception as e:
        message = f"Error checking cuDNN availability: {e}"
        return False, message


def check_backend(os_name):
    """Check if the appropriate distributed backend is available based on the OS."""
    try:
        if os_name == "Windows":
            # Check Gloo backend
            gloo_available = dist.is_gloo_available()
            if gloo_available:
                message = "Gloo backend is available."
                return True, "gloo", message
            else:
                message = "Gloo backend is not available."
                return False, None, message
        else:
            # Check NCCL backend
            nccl_available = dist.is_nccl_available()
            if nccl_available:
                message = "NCCL backend is available."
                return True, "nccl", message
            else:
                message = "NCCL backend is not available."
                return False, None, message
    except Exception as e:
        message = f"Error checking distributed backend: {e}"
        return False, None, message


def check_multiprocessing():
    """Check if multiprocessing is available."""
    try:
        ctx = mp.get_context("spawn")
        message = "Multiprocessing is available."
        return True, message
    except RuntimeError as e:
        message = f"Multiprocessing is not available: {e}"
        return False, message


def check_gpu():
    """Check the number of GPUs and their names."""
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            message = f"{gpu_count} GPU(s) available: {gpu_names}"
            return True, message
        else:
            message = "No GPUs are available."
            return False, message
    except Exception as e:
        message = f"Error checking GPUs: {e}"
        return False, message


def run(rank, size, backend):
    """Function to be run by each process."""
    try:
        # Set the appropriate GPU for this rank
        torch.cuda.set_device(rank)
        # Initialize the process group
        dist.init_process_group(
            backend=backend, init_method="env://", rank=rank, world_size=size
        )
        torch.manual_seed(0)
        tensor = torch.tensor([1.0]).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # No need to print the result
        dist.destroy_process_group()
    except Exception as e:
        print(f"Process {rank} encountered an error: {e}")


def check_distributed(os_name, backend, gpu_count):
    """Test distributed training."""
    try:
        distributed_available = dist.is_available()
        if not distributed_available:
            message = "torch.distributed is not available."
            return False, message

        if gpu_count < 2:
            message = (
                "torch.distributed is available, but less than 2 GPUs are detected."
            )
            return True, message

        # Check backend
        if backend is None:
            message = "No suitable backend is available. Cannot perform distributed tensor operations."
            return False, message

        try:
            mp.spawn(run, args=(gpu_count, backend), nprocs=gpu_count, join=True)
            message = (
                f"Distributed tensor operations succeeded using {backend} backend."
            )
            return True, message
        except Exception as e:
            message = f"Distributed tensor operations failed: {e}"
            return False, message
    except Exception as e:
        message = f"Error in distributed check: {e}"
        return False, message


def main():
    # Parse command-line arguments
    master_addr, master_port, dependencies_file = parse_args()

    # Set environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    results = {}

    # Check OS
    os_name, message_os = check_os()
    results["OS Information"] = {"Success": True, "Message": message_os}

    # Check Python version
    success, message = check_python_version()
    results["Python Version"] = {"Success": success, "Message": message}

    # Check dependencies
    success_deps, message_deps = check_dependencies(dependencies_file)
    results["Dependencies"] = {"Success": success_deps, "Message": message_deps}

    if not success_deps:
        # If dependencies are insufficient, skip the rest of the checks
        generate_report(results)
        sys.exit(1)

    # Check PyTorch
    success, message = check_pytorch()
    results["PyTorch Installation and Version"] = {
        "Success": success,
        "Message": message,
    }

    if success:
        # PyTorch is installed
        # Check CUDA
        success_cuda, message_cuda = check_cuda()
        results["CUDA Availability and Version"] = {
            "Success": success_cuda,
            "Message": message_cuda,
        }

        if success_cuda:
            # If CUDA is available, check cuDNN
            success_cudnn, message_cudnn = check_cudnn()
            results["cuDNN Availability and Version"] = {
                "Success": success_cudnn,
                "Message": message_cudnn,
            }
        else:
            # If CUDA is not available, skip cuDNN check
            results["cuDNN Availability and Version"] = {
                "Success": False,
                "Message": "CUDA not available; skipping cuDNN check.",
            }

        # Check multiprocessing
        success_mp, message_mp = check_multiprocessing()
        results["Multiprocessing Availability"] = {
            "Success": success_mp,
            "Message": message_mp,
        }

        # Check GPUs
        success_gpu, message_gpu = check_gpu()
        results["GPU Count and Names"] = {
            "Success": success_gpu,
            "Message": message_gpu,
        }
        gpu_count = torch.cuda.device_count() if success_gpu else 0

        # Check distributed backend
        backend_available, backend_name, backend_message = check_backend(os_name)
        results["Distributed Backend Availability"] = {
            "Success": backend_available,
            "Message": backend_message,
        }

        # Test distributed tensor operations
        success_distributed, message_distributed = check_distributed(
            os_name, backend_name, gpu_count
        )
        results["torch.distributed and Tensor Operations"] = {
            "Success": success_distributed,
            "Message": message_distributed,
        }

    else:
        # If PyTorch is not installed, skip the rest of the checks
        results["CUDA Availability and Version"] = {
            "Success": False,
            "Message": "PyTorch not installed; skipping CUDA check.",
        }
        results["cuDNN Availability and Version"] = {
            "Success": False,
            "Message": "PyTorch not installed; skipping cuDNN check.",
        }
        results["Multiprocessing Availability"] = {
            "Success": False,
            "Message": "PyTorch not installed; skipping multiprocessing check.",
        }
        results["GPU Count and Names"] = {
            "Success": False,
            "Message": "PyTorch not installed; skipping GPU check.",
        }
        results["Distributed Backend Availability"] = {
            "Success": False,
            "Message": "PyTorch not installed; skipping backend check.",
        }
        results["torch.distributed and Tensor Operations"] = {
            "Success": False,
            "Message": "PyTorch not installed; skipping torch.distributed check.",
        }

    # Generate the report
    generate_report(results)


def generate_report(results):
    """Print the verification results in an intuitive format."""
    print("\n====== PyTorch Environment Check Report ======\n")
    for name, result in results.items():
        status = "✅" if result["Success"] else "❌"
        print(f"[{status}] {name}: {result['Message']}")
    print("\n=============================================\n")


if __name__ == "__main__":
    # Only execute if the script is run directly
    mp.set_start_method("spawn", force=True)
    main()
