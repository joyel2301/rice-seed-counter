"""
Developer tool: convert best.pt to OpenVINO models (needs PyTorch + Ultralytics; the app itself does not).

  python scripts/convert_openvino.py          # make missing files only
  python scripts/convert_openvino.py --force  # remake all

For every input size (640 / 960 / 1280) three fixed shapes are made, so any picture can be fed
at full resolution: landscape 16:9, square, and portrait 9:16. Output:
external_models/rice_grain_phenotyping/openvino/best_<H>x<W>.xml/.bin (FP32 weights).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from ov_backend import OV_SHAPES  # noqa: E402


def main(force=False):
    import openvino as ov
    import torch
    import extra_modules_compat  # noqa: F401  (layers needed to load best.pt)
    from ultralytics import YOLO

    out = ROOT / "external_models/rice_grain_phenotyping/openvino"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "external_models/rice_grain_phenotyping/best.pt")).model.float().eval().fuse()
    for p in model.parameters():
        p.requires_grad = False
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, "openvino", False   # raw (1, 5, N) output, no NMS

    shapes = sorted({s for group in OV_SHAPES.values() for s in group})
    for h, w in shapes:
        xml = out / f"best_{h}x{w}.xml"
        if xml.exists() and not force:
            continue
        t = time.time()
        x = torch.zeros(1, 3, h, w)
        with torch.no_grad():
            model(x)   # let the Detect head cache its anchors for this shape, or tracing fails
            ov_model = ov.convert_model(model, example_input=x, input=[1, 3, h, w])
        ov.save_model(ov_model, str(xml), compress_to_fp16=False)
        print(f"{xml.name}  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
