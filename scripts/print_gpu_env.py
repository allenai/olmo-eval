"""Print a summary of runtime dependencies at container startup.

This script is invoked via `uv run python` to use the uv-managed venv.
"""

import sys


def main() -> None:
    print("\n" + "=" * 60)
    print("RUNTIME ENVIRONMENT SUMMARY")
    print("=" * 60)

    # Python version
    print(f"Python:          {sys.version.split()[0]}")

    # PyTorch and CUDA
    try:
        import torch

        print(f"PyTorch:         {torch.__version__}")
        print(f"CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version:    {torch.version.cuda}")
            print(f"cuDNN version:   {torch.backends.cudnn.version()}")
            print(f"GPU count:       {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}:         {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch:         NOT INSTALLED")

    # Flash Attention
    try:
        import flash_attn

        print(f"Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("Flash Attention: NOT INSTALLED")

    # Transformers
    try:
        import transformers

        print(f"Transformers:    {transformers.__version__}")
    except ImportError:
        print("Transformers:    NOT INSTALLED")

    # vLLM
    try:
        import vllm

        print(f"vLLM:            {vllm.__version__}")
    except ImportError:
        print("vLLM:            NOT INSTALLED")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
