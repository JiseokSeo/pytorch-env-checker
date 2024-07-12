import sys
import os


def check_python_version():
    try:
        python_version = sys.version
        print(f"✅ Python version: {python_version}")
    except Exception as e:
        print(f"❌ An error occurred while checking Python version: {e}")


def check_gpu_info():
    try:
        gpu_info = (
            os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
            .read()
            .strip()
        )
        if gpu_info:
            print(f"✅ GPU info:\n{gpu_info}")
        else:
            print("❌ GPU info could not be determined.")
    except Exception as e:
        print(f"❌ An error occurred while checking GPU info: {e}")


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


def check_device_info():
    try:
        cpu_info = (
            os.popen(
                "lscpu | grep 'Model name\|Socket(s)\|Core(s) per socket\|Thread(s) per core'"
            )
            .read()
            .strip()
        )
        print(f"✅ CPU info:\n{cpu_info}")
    except Exception as e:
        print(f"❌ An error occurred while checking CPU info: {e}")


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


def check_pytorch_version():
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch is not installed.")
    except Exception as e:
        print(f"❌ An error occurred while checking PyTorch version: {e}")


def check_cpp_extensions():
    try:
        import torch

        if torch.utils.cpp_extension.is_available():
            print("✅ PyTorch C++ extensions are available.")
        else:
            print("❌ PyTorch C++ extensions are not available.")
    except Exception as e:
        print(f"❌ An error occurred while checking PyTorch C++ extensions: {e}")


def check_cpp_extension_version():
    try:
        import torch

        if torch.utils.cpp_extension.is_available():
            cpp_ext_version = torch.utils.cpp_extension.version()
            print(f"✅ PyTorch C++ extensions version: {cpp_ext_version}")
        else:
            print("❌ PyTorch C++ extensions are not available.")
    except Exception as e:
        print(
            f"❌ An error occurred while checking PyTorch C++ extensions version: {e}"
        )


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


def check_cuda_version():
    try:
        import torch

        if hasattr(torch.version, "cuda"):
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("❌ CUDA version could not be determined.")
    except Exception as e:
        print(f"❌ An error occurred while checking CUDA version: {e}")


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

    print("🛠 Checking GPU Info:")
    check_gpu_info()
    print("\n")

    print("🛠 Checking GPU Driver Version:")
    check_gpu_driver_version()
    print("\n")

    print("🛠 Checking Device Info:")
    check_device_info()
    print("\n")

    print("🛠 Checking Memory Usage:")
    check_memory_usage()
    print("\n")

    print("🛠 Checking PyTorch Version:")
    check_pytorch_version()
    print("\n")

    print("🛠 Checking PyTorch C++ Extensions:")
    check_cpp_extensions()
    print("\n")

    print("🛠 Checking PyTorch C++ Extensions Version:")
    check_cpp_extension_version()
    print("\n")

    print("🛠 Checking CUDA Availability:")
    check_cuda()
    print("\n")

    print("🛠 Checking CUDA Version:")
    check_cuda_version()
    print("\n")

    print("🛠 Checking cuDNN Availability:")
    check_cudnn()
    print("\n")

    print("🛠 Checking NCCL Availability:")
    check_nccl()
    print("\n")

    print("🛠 Checking Distributed Backend Availability:")
    check_distributed_backend()
    print("\n")
