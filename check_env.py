import sys
import os


def check_python_version():
    try:
        python_version = sys.version
        print(f"✅ Python version: {python_version}")
    except Exception as e:
        print(f"❌ An error occurred while checking Python version: {e}")


def check_gpu_driver_version():
    try:
        gpu_driver_version = (
            os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
            .read()
            .strip()
        )
        if gpu_driver_version:
            print(f"✅ GPU driver version: {gpu_driver_version}")
        else:
            print("❌ GPU driver version could not be determined.")
    except Exception as e:
        print(f"❌ An error occurred while checking GPU driver version: {e}")


def check_pytorch_version():
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch is not installed.")
    except Exception as e:
        print(f"❌ An error occurred while checking PyTorch version: {e}")


def check_cuda_version():
    try:
        import torch

        if hasattr(torch.version, "cuda"):
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("❌ CUDA version could not be determined.")
    except Exception as e:
        print(f"❌ An error occurred while checking CUDA version: {e}")


def check_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            print("✅ CUDA is available.")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - Memory allocated: {torch.cuda.memory_allocated(i)} bytes")
                print(f"  - Memory reserved: {torch.cuda.memory_reserved(i)} bytes")
        else:
            print("❌ CUDA is not available.")
    except Exception as e:
        print(f"❌ An error occurred while checking CUDA availability: {e}")


def check_cudnn():
    try:
        import torch

        if torch.backends.cudnn.is_available():
            print("✅ cuDNN is available.")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        else:
            print("❌ cuDNN is not available.")
    except Exception as e:
        print(f"❌ An error occurred while checking cuDNN availability: {e}")


def check_nccl():
    try:
        import torch

        if hasattr(torch.cuda, "nccl") and torch.cuda.nccl.is_available():
            print("✅ NCCL is available.")
        else:
            print("❌ NCCL is not available.")
    except Exception as e:
        print(f"❌ An error occurred while checking NCCL availability: {e}")


def check_torch_distributed():
    try:
        import torch

        if torch.distributed.is_available():
            print("✅ torch.distributed is available.")
        else:
            print("❌ torch.distributed is not available.")
    except Exception as e:
        print(
            f"❌ An error occurred while checking torch.distributed availability: {e}"
        )


def check_torch_configuration():
    try:
        import torch

        num_threads = torch.get_num_threads()
        print(f"✅ PyTorch is configured to use {num_threads} threads.")
    except Exception as e:
        print(f"❌ An error occurred while checking PyTorch configuration: {e}")


def check_memory_usage():
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} Memory Usage:")
                print(f"  - Allocated: {torch.cuda.memory_allocated(i)} bytes")
                print(f"  - Reserved: {torch.cuda.memory_reserved(i)} bytes")
        else:
            print("❌ CUDA is not available. Cannot check GPU memory usage.")
    except Exception as e:
        print(f"❌ An error occurred while checking memory usage: {e}")


def check_device_properties():
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i} Properties:")
                print(f"  - Name: {props.name}")
                print(f"  - Multiprocessors: {props.multi_processor_count}")
                print(f"  - Max Threads Per Block: {props.max_threads_per_block}")
                print(
                    f"  - Shared Memory Per Block: {props.shared_memory_per_block} bytes"
                )
        else:
            print("❌ CUDA is not available. Cannot check device properties.")
    except Exception as e:
        print(f"❌ An error occurred while checking device properties: {e}")


def check_cpp_extensions():
    try:
        import torch

        if torch.utils.cpp_extension.is_available():
            print("✅ PyTorch C++ extensions are available.")
        else:
            print("❌ PyTorch C++ extensions are not available.")
    except Exception as e:
        print(f"❌ An error occurred while checking PyTorch C++ extensions: {e}")


def check_distributed_backend():
    try:
        import torch

        available_backends = []
        if torch.distributed.is_available():
            if torch.distributed.is_nccl_available():
                available_backends.append("NCCL")
            if torch.distributed.is_gloo_available():
                available_backends.append("Gloo")
            if torch.distributed.is_mpi_available():
                available_backends.append("MPI")
            if available_backends:
                print(
                    f"✅ Available distributed backends: {', '.join(available_backends)}"
                )
            else:
                print("❌ No distributed backends are available.")
        else:
            print("❌ torch.distributed is not available.")
    except Exception as e:
        print(
            f"❌ An error occurred while checking distributed backend availability: {e}"
        )


def check_all():
    print("🔍 Checking PyTorch environment settings...\n")

    print("🛠 Checking Python Version:")
    check_python_version()
    print("\n")

    print("🛠 Checking GPU Driver Version:")
    check_gpu_driver_version()
    print("\n")

    print("🛠 Checking PyTorch Version:")
    check_pytorch_version()
    print("\n")

    print("🛠 Checking CUDA Version:")
    check_cuda_version()
    print("\n")

    print("🛠 Checking CUDA Availability:")
    check_cuda()
    print("\n")

    print("🛠 Checking cuDNN Availability:")
    check_cudnn()
    print("\n")

    print("🛠 Checking NCCL Availability:")
    check_nccl()
    print("\n")

    print("🛠 Checking torch.distributed Availability:")
    check_torch_distributed()
    print("\n")

    print("🛠 Checking PyTorch Configuration:")
    check_torch_configuration()
    print("\n")

    print("🛠 Checking Memory Usage:")
    check_memory_usage()
    print("\n")

    print("🛠 Checking Device Properties:")
    check_device_properties()
    print("\n")

    print("🛠 Checking PyTorch C++ Extensions:")
    check_cpp_extensions()
    print("\n")

    print("🛠 Checking Distributed Backend Availability:")
    check_distributed_backend()
    print("\n")


if __name__ == "__main__":
    check_all()
