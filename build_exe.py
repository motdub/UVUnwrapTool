"""build_exe.py - run on WINDOWS with Python installed (build machine only).
Creates UVUnwrapTool_exe.zip: unzip anywhere and run UVUnwrapTool.exe.
No Python, pip or internet needed on the machine that runs it.

Usage:  python build_exe.py      (keep app.py and uvcore.py in the same folder)"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "UVUnwrapTool"

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
                       "numpy", "trimesh", "xatlas", "pyinstaller"])

for d in ("build", "dist"):
    shutil.rmtree(HERE / d, ignore_errors=True)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm", "--clean", "--onedir", "--windowed",
    "--name", NAME,
    "--hidden-import", "uvcore",
    "--collect-all", "xatlas",
    "--collect-all", "trimesh",
]
# Optional heavy packages we never use; excluding them shrinks the exe and its startup.
for mod in ("matplotlib", "scipy", "pandas", "IPython", "pytest", "PIL", "networkx", "shapely"):
    cmd += ["--exclude-module", mod]
cmd.append(str(HERE / "app.py"))
subprocess.check_call(cmd, cwd=HERE)

zip_path = shutil.make_archive(str(HERE / f"{NAME}_exe"), "zip", root_dir=HERE / "dist", base_dir=NAME)
print(f"\nDone: {zip_path}\nRun {NAME}/{NAME}.exe after unzipping.")
