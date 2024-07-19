# pytorch-env-checker
Comprehensive toolkit for diagnosing and validating your PyTorch environment, ensuring compatibility and optimal performance.

## Features

- **Python Version Check:** Ensures the Python version is compatible with PyTorch.
- **GPU Driver Version Check:** Checks if the GPU driver version is compatible with CUDA and PyTorch.
- **PyTorch Version Check:** Verifies the installed PyTorch version.
- **CUDA Version Check:** Confirms the CUDA version.
- **CUDA Availability Check:** Checks if CUDA is available and provides details on available GPUs.
- **cuDNN Availability Check:** Verifies if cuDNN is available and its version.
- **NCCL Availability Check:** Ensures NCCL is available for distributed computing.
- **torch.distributed Availability Check:** Verifies the availability of `torch.distributed`.
- **PyTorch Configuration Check:** Checks PyTorch configuration settings like the number of threads.
- **Memory Usage Check:** Monitors CPU and GPU memory usage.
- **Device Properties Check:** Provides detailed properties of CUDA devices.
- **PyTorch C++ Extensions Check:** Ensures custom C++ extensions for PyTorch are correctly compiled and loaded.
- **Distributed Backend Check:** Checks configurations for distributed training backends like NCCL, Gloo, or MPI.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/JiseokSeo/pytorch-env-checker.git
    cd pytorch-env-checker
    ```

2. Run the script:
    ```bash
    python check_env.py
    ```

## License

This project is licensed under the MIT License.



# PyTorch Env Checker

PyTorch 환경을 진단하고 검증하기 위한 종합 도구 키트입니다.

## 기능

- **Python 버전 확인:** Python 버전이 PyTorch와 호환되는지 확인합니다.
- **GPU 드라이버 버전 확인:** GPU 드라이버 버전이 CUDA 및 PyTorch와 호환되는지 확인합니다.
- **PyTorch 버전 확인:** 설치된 PyTorch 버전을 확인합니다.
- **CUDA 버전 확인:** CUDA 버전을 확인합니다.
- **CUDA 사용 가능 여부 확인:** CUDA 사용 가능 여부를 확인하고 사용 가능한 GPU의 세부 정보를 제공합니다.
- **cuDNN 사용 가능 여부 확인:** cuDNN 사용 가능 여부와 버전을 확인합니다.
- **NCCL 사용 가능 여부 확인:** 분산 컴퓨팅을 위한 NCCL 사용 가능 여부를 확인합니다.
- **torch.distributed 사용 가능 여부 확인:** `torch.distributed` 사용 가능 여부를 확인합니다.
- **PyTorch 설정 확인:** PyTorch 설정(예: 스레드 수)을 확인합니다.
- **메모리 사용량 확인:** CPU 및 GPU 메모리 사용량을 모니터링합니다.
- **장치 속성 확인:** CUDA 장치의 자세한 속성을 제공합니다.
- **PyTorch C++ 확장 확인:** PyTorch를 위한 사용자 정의 C++ 확장이 올바르게 컴파일되고 로드되었는지 확인합니다.
- **분산 백엔드 확인:** NCCL, Gloo 또는 MPI와 같은 분산 학습 백엔드 구성을 확인합니다.

## 사용법

1. 레포지토리 클론:
    ```bash
    git clone https://github.com/JiseokSeo/pytorch-env-checker.git
    cd pytorch-env-checker
    ```

2. 스크립트 실행:
    ```bash
    python check_env.py
    ```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

