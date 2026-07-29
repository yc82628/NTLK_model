"""
check_gpu.py — Run this FIRST before anything else.

Verifies your NVIDIA GPU, CUDA drivers, and PyTorch GPU support
are all correctly configured before you invest time setting up
the full pipeline.

Usage:
    python check_gpu.py
"""

import subprocess
import sys


def check_nvidia_driver():
    print("=" * 60)
    print("1. Checking NVIDIA driver (nvidia-smi)...")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
            print("✓ NVIDIA driver detected and working.\n")
            return True
        else:
            print("✗ nvidia-smi ran but returned an error.")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found. This usually means:")
        print("    - No NVIDIA GPU installed, OR")
        print("    - NVIDIA drivers are not installed")
        print("    Fix: install drivers from nvidia.com/drivers")
        return False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi timed out.")
        return False


def check_torch_cuda():
    print("=" * 60)
    print("2. Checking PyTorch CUDA support...")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed yet.")
        print("    Fix: pip install torch (see setup instructions)")
        return False

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version (PyTorch built with): {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")
        print("\n✓ PyTorch can see and use your GPU.\n")

        # Rough VRAM guidance for this pipeline
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram < 4:
            print("⚠ WARNING: Less than 4GB VRAM detected.")
            print("  DistilBERT fine-tuning may fail with CUDA out-of-memory errors.")
            print("  Reduce BATCH_SIZE to 4 or 8 in config.py if you hit OOM errors.")
        elif total_vram < 6:
            print("⚠ NOTE: 4-6GB VRAM. Use BATCH_SIZE=8 to be safe.")
        else:
            print("✓ VRAM looks sufficient for BATCH_SIZE=16 (the pipeline default).")
        return True
    else:
        print("\n✗ PyTorch was installed WITHOUT CUDA support, or no GPU found.")
        print("  This is the most common setup mistake. Fix:")
        print("  1. Uninstall current torch: pip uninstall torch")
        print("  2. Reinstall with CUDA support — go to https://pytorch.org/get-started/locally/")
        print("     and copy the exact install command for your CUDA version.")
        print("     Example for CUDA 12.1:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False


def main():
    print("\nGPU ENVIRONMENT CHECK FOR STEAM NLP PIPELINE\n")

    driver_ok = check_nvidia_driver()
    torch_ok = check_torch_cuda()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if driver_ok and torch_ok:
        print("✓ All checks passed. You're ready to run the pipeline.")
        sys.exit(0)
    else:
        print("✗ Some checks failed. Fix the issues above before proceeding.")
        print("  If you don't have an NVIDIA GPU, the pipeline will still")
        print("  run on CPU, but DistilBERT fine-tuning will take hours")
        print("  instead of minutes. Consider Kaggle Notebooks instead.")
        sys.exit(1)


if __name__ == "__main__":
    main()
