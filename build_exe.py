#!/usr/bin/env python
"""Build a portable, single-file Windows .exe for app.py using PyInstaller.

Usage:
    python build_exe.py

Produces dist/BaySpecReflectanceViewer.exe — a standalone executable that
runs on a machine with no Python install.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "BaySpecReflectanceViewer"


def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found; installing into this environment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)


def build():
    ensure_pyinstaller()
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", APP_NAME,
        "--collect-data", "matplotlib",
        "--hidden-import", "matplotlib.backends.backend_tkagg",
        str(ROOT / "app.py"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    exe_path = ROOT / "dist" / f"{APP_NAME}.exe"
    print(f"\nBuild complete: {exe_path}")


if __name__ == "__main__":
    build()
