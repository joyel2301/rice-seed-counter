# PyInstaller build for the Rice Seed Counter GUI (folder build, no Python needed on the target PC).
#   riceseedcounter-env\python.exe -m PyInstaller RiceSeedCounter.spec --noconfirm
# Output: dist\RiceSeedCounter\RiceSeedCounter.exe  (zip the whole RiceSeedCounter folder)
#
# The app runs on OpenVINO only, so PyTorch / Ultralytics are left out of the bundle.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)
OV_DIR = ROOT / "external_models/rice_grain_phenotyping/openvino"

datas = [(str(ROOT / "assets/app_icon.ico"), "assets")]
# OpenVINO model files only; the GPU kernel cache is driver-specific and is rebuilt on each PC.
for f in OV_DIR.glob("best_*"):
    datas.append((str(f), "external_models/rice_grain_phenotyping/openvino"))
datas += collect_data_files("openvino")
binaries = collect_dynamic_libs("openvino")         # device plugins (GPU/CPU)

a = Analysis(
    [str(ROOT / "scripts/rice_counter.py")],
    pathex=[str(ROOT / "scripts")],
    binaries=binaries,
    datas=datas,
    hiddenimports=["openvino.opset13", "openvino.preprocess"],
    excludes=["torch", "torchvision", "ultralytics", "timm", "scipy", "pandas", "matplotlib",
              "polars", "sympy", "IPython", "jupyter", "notebook", "tensorboard", "pytest", "tkinter"],
    noarchive=False,
)

# Drop large files the app never uses (it only reads OpenVINO IR on CPU/GPU and webcams via DirectShow/MSMF).
UNUSED = (
    "openvino_intel_npu_compiler.dll",                 # 96 MB, NPU device not used
    "_frontend.dll",                                   # ONNX/TF/PyTorch/Paddle/JAX/GGUF readers (IR reader kept below)
    "openvino.lib", "_frontend.lib",                   # link libraries
    "opencv_videoio_ffmpeg",                           # 27 MB, video-file decoding
    "opengl32sw.dll",                                  # 20 MB, Qt software OpenGL
)


def keep(entry):
    name = entry[0].replace("\\", "/").lower()
    if "openvino_ir_frontend" in name:
        return True
    if "pyside6/translations/" in name:
        return False
    return not any(u.lower() in name for u in UNUSED)


a.binaries = [b for b in a.binaries if keep(b)]
a.datas = [d for d in a.datas if keep(d)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RiceSeedCounter",
    console=False,
    icon=str(ROOT / "assets/app_icon.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="RiceSeedCounter")
