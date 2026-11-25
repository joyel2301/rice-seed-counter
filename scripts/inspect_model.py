"""
Look inside the rice-grain model file (best.pt).

  python scripts/inspect_model.py            # everything below
  python scripts/inspect_model.py classes    # 1. classes the file needs (read without running any code)
  python scripts/inspect_model.py info       # 2. training settings stored in the file
  python scripts/inspect_model.py layers     # 3. top-level layers with parameter counts
  python scripts/inspect_model.py tree       # 4. full layer tree (long)
  python scripts/inspect_model.py weights    # 5. every weight tensor and its shape (long)

Output can be saved with:  ... > model_info.txt
"""
import pickletools
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "external_models/rice_grain_phenotyping/best.pt"


def show_classes():
    """List every class the pickle refers to, by reading its opcodes only (nothing is executed)."""
    print("=" * 70)
    print("1. 파일이 필요로 하는 클래스 (코드 실행 없이 읽음)")
    print("=" * 70)
    with zipfile.ZipFile(MODEL) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        data = z.read(name)
    strings, refs = [], Counter()
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            refs[f"{strings[-2]}.{strings[-1]}"] += 1
        elif op.name == "GLOBAL":
            refs[arg.replace(" ", ".")] += 1
    for group, title in [("extra_modules", "연구팀이 추가한 레이어 (일반 Ultralytics에 없음)"),
                         ("timm", "timm 라이브러리 레이어"),
                         ("ultralytics", "일반 Ultralytics 레이어"),
                         ("torch", "PyTorch 기본 레이어")]:
        items = sorted((k, v) for k, v in refs.items() if group in k)
        if group == "ultralytics":
            items = [(k, v) for k, v in items if "extra_modules" not in k]
        if items:
            print(f"\n[{title}]")
            for k, _ in items:     # pickle stores each class once, so no usage counts here
                print(f"  {k}")
    print()


def load():
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import extra_modules_compat  # noqa: F401  (re-implemented layers, see that file)
    return torch.load(MODEL, map_location="cpu", weights_only=False)


def show_info(ckpt):
    print("=" * 70)
    print("2. 파일에 저장된 학습 정보")
    print("=" * 70)
    print(f"  저장 날짜        : {ckpt.get('date')}")
    print(f"  Ultralytics 버전 : {ckpt.get('version')}")
    print(f"  학습 epoch       : {ckpt.get('epoch')}")
    a = ckpt.get("train_args", {})
    for k in ["model", "data", "imgsz", "epochs", "batch", "optimizer", "lr0", "iou", "max_det"]:
        print(f"  {k:16s} : {a.get(k)}")
    m = ckpt["model"]
    print(f"  클래스 이름      : {m.names}")
    print(f"  출력 stride      : {m.stride.tolist()}  (4 = 이미지를 4배만 줄인 고해상도 지도에서 검출)")
    n = sum(p.numel() for p in m.parameters())
    print(f"  전체 파라미터    : {n:,}")
    print()


def show_layers(model):
    print("=" * 70)
    print("3. 레이어 순서 (from = 입력을 받는 앞 레이어 번호, -1 = 바로 앞)")
    print("=" * 70)
    print(f"  {'#':>3}  {'from':>10}  {'파라미터':>10}  레이어")
    for i, layer in enumerate(model.model):
        n = sum(p.numel() for p in layer.parameters())
        print(f"  {i:>3}  {str(layer.f):>10}  {n:>10,}  {type(layer).__name__}")
    print()


def show_tree(model):
    print("=" * 70)
    print("4. 전체 레이어 구조")
    print("=" * 70)
    print(model)
    print()


def show_weights(model):
    print("=" * 70)
    print("5. 가중치 텐서와 모양")
    print("=" * 70)
    for name, t in model.state_dict().items():
        print(f"  {name:70s} {tuple(t.shape)}")
    print()


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("classes", "all"):
        show_classes()
    if what == "classes":
        return
    ckpt = load()
    model = ckpt["model"].float()
    if what in ("info", "all"):
        show_info(ckpt)
    if what in ("layers", "all"):
        show_layers(model)
    if what in ("tree", "all"):
        show_tree(model)
    if what in ("weights", "all"):
        show_weights(model)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
