"""
Non-GUI parts of the rice seed counter: engine loading, detection, false-detection filter,
region-of-interest, automatic input size, camera discovery, image I/O.

Runs on OpenVINO only (no PyTorch). The detection model is best.pt from
BearHuntingExpert/Rice-Grain-Phenotyping-System (MIT), converted to OpenVINO with
scripts/convert_openvino.py.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from camera_names import is_windows_virtual_camera, list_msmf_video_devices, list_video_devices
from ov_backend import OVDetector, best_shape, openvino_available, openvino_devices

OV_DIR = "external_models/rice_grain_phenotyping/openvino"
MAX_DETECTIONS = 3000          # Ultralytics' default (300) would cap the count

ENGINE_OV_GPU = "OpenVINO GPU FP16 (빠름)"
ENGINE_OV_GPU32 = "OpenVINO GPU FP32 (정밀)"
ENGINE_OV_CPU = "OpenVINO CPU"
OV_ENGINES = {ENGINE_OV_GPU: ("GPU", "f16"), ENGINE_OV_GPU32: ("GPU", "f32"), ENGINE_OV_CPU: ("CPU", "f32")}


# ---------------------------------------------------------------- paths
def bundle_path(relative):
    """File shipped with the app (inside the PyInstaller bundle, or the project folder)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def licenses_dir():
    """License texts: next to the .exe when packaged, else <project>/licenses."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "licenses"
    return Path(__file__).resolve().parent.parent / "licenses"


def output_dir():
    """Where CSVs and photos go: next to the .exe when packaged, else <project>/results."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "results"
    return Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------- engines
def available_engines():
    if not openvino_available(bundle_path(OV_DIR)):
        return []
    engines = [ENGINE_OV_GPU, ENGINE_OV_GPU32] if "GPU" in openvino_devices() else []
    return engines + [ENGINE_OV_CPU]


def load_detector(engine, imgsz=1280):
    """Return (detector, engine actually used). If the GPU fails (e.g. no Intel GPU driver), use the CPU."""
    device, precision = OV_ENGINES.get(engine, OV_ENGINES[ENGINE_OV_CPU])
    try:
        det = OVDetector(bundle_path(OV_DIR), device, precision=precision)
        det.warm_up(imgsz)
        return det, engine
    except Exception as e:
        if device == "CPU":
            raise
        print(f"OpenVINO {device} failed, using CPU: {e}")
        import app_log
        app_log.log(f"OpenVINO {device} ({precision}) failed, falling back to CPU: {e!r}")
    det = OVDetector(bundle_path(OV_DIR), "CPU")
    det.warm_up(imgsz)
    return det, ENGINE_OV_CPU


def load_photo_detector():
    """Photos are analysed on the CPU in FP32: exact, and no GPU compile wait for new shapes."""
    return OVDetector(bundle_path(OV_DIR), "CPU")


def detect(detector, frame, conf, iou, imgsz):
    """Return (boxes Nx4 xyxy float32, scores N) in frame pixels."""
    return detector.detect(frame, conf=conf, iou=iou, imgsz=imgsz, max_det=MAX_DETECTIONS)


# ---------------------------------------------------------------- filtering
def filter_detections(frame, xyxy, min_saturation=40, size_lo=0.25, size_hi=4.0):
    """
    Keep-mask that drops boxes whose centre is nearly colourless (paper, text, shadows;
    labelled grains have centre saturation >= 79) and, with 5+ boxes, boxes far smaller or
    larger than the median box.
    """
    if len(xyxy) == 0:
        return np.zeros(0, dtype=bool)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]
    keep = np.ones(len(xyxy), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        w, h = x2 - x1, y2 - y1
        a1 = int(np.clip(x1 + w / 4, 0, w_img - 1)); a2 = int(np.clip(x2 - w / 4, a1 + 1, w_img))
        b1 = int(np.clip(y1 + h / 4, 0, h_img - 1)); b2 = int(np.clip(y2 - h / 4, b1 + 1, h_img))
        if hsv[b1:b2, a1:a2, 1].mean() < min_saturation:
            keep[i] = False
    if keep.sum() >= 5:
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        median = np.median(area[keep])
        keep &= (area >= size_lo * median) & (area <= size_hi * median)
    return keep


def roi_to_pixels(roi, shape):
    """roi = (x, y, w, h) as fractions of the frame -> (x1, y1, x2, y2) pixels, or None."""
    if not roi:
        return None
    h, w = shape[:2]
    x, y, rw, rh = roi
    return (int(round(x * w)), int(round(y * h)), int(round((x + rw) * w)), int(round((y + rh) * h)))


def inside_roi(xyxy, roi_px):
    """Keep-mask: box centre inside the region."""
    if roi_px is None or len(xyxy) == 0:
        return np.ones(len(xyxy), dtype=bool)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    x1, y1, x2, y2 = roi_px
    return (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)


# ---------------------------------------------------------------- input size
SIZES = (640, 960, 1280)
# The model was trained on 3024 px photos fed at 640, where a grain is ~24-30 px long inside the
# network. Much bigger (e.g. 60 px: the same photos fed at 1280) and grains start to split into two
# boxes; much smaller (13 px: HD video fed at 640) and grains are missed.
TARGET_GRAIN_PX = 28


def recommend_imgsz(boxes, frame_shape):
    """Input size that makes the median grain ~TARGET_GRAIN_PX long inside the network (None if < 5 grains)."""
    if len(boxes) < 5:
        return None
    side = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))
    if side <= 0:
        return None

    def miss(size):
        h, w = best_shape(frame_shape, size)
        r = min(h / frame_shape[0], w / frame_shape[1])
        return abs(np.log(side * r / TARGET_GRAIN_PX))

    return min(SIZES, key=miss)


# ---------------------------------------------------------------- analysis of one frame
ROI_MARGIN = 0.03   # crop this much (fraction of the frame) beyond the region so edge grains stay whole


def crop_for_roi(frame, roi_px):
    """(crop, (ox, oy)): the region plus a small margin, or the whole frame when there is no region."""
    if roi_px is None:
        return frame, (0, 0)
    h, w = frame.shape[:2]
    m = int(round(ROI_MARGIN * max(h, w)))
    x1, y1, x2, y2 = roi_px
    x1, y1 = max(0, x1 - m), max(0, y1 - m)
    x2, y2 = min(w, x2 + m), min(h, y2 + m)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return frame, (0, 0)
    return frame[y1:y2, x1:x2], (x1, y1)


class Analysis:
    """Result for one frame. input_shape = shape of the picture the model saw (the region crop)."""
    __slots__ = ("frame", "boxes", "scores", "roi_px", "imgsz", "input_shape")

    def __init__(self, frame, boxes, scores, roi_px=None, imgsz=None, input_shape=None):
        self.frame, self.boxes, self.scores = frame, boxes, scores
        self.roi_px, self.imgsz = roi_px, imgsz
        self.input_shape = input_shape if input_shape is not None else frame.shape

    @property
    def count(self):
        return len(self.boxes)


def analyze(detector, frame, s, imgsz=None):
    """
    s: dict with conf, iou, imgsz (int or "auto"), filter (bool), roi (fractions or None).
    imgsz overrides s["imgsz"] (callers resolve "auto" with recommend_imgsz / AutoSize).

    With a region, only the region (plus a small margin) is given to the model, so its full input
    resolution goes to the grains instead of the empty parts of the picture. Boxes are moved back to
    frame coordinates and only those with their centre inside the region are kept.
    """
    size = imgsz or (s["imgsz"] if s["imgsz"] != "auto" else 960)
    roi_px = roi_to_pixels(s.get("roi"), frame.shape)
    crop, (ox, oy) = crop_for_roi(frame, roi_px)
    boxes, scores = detect(detector, crop, s["conf"], s["iou"], size)
    if ox or oy:
        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy
    keep = inside_roi(boxes, roi_px)
    if s.get("filter") and len(boxes):
        keep &= filter_detections(frame, boxes)
    return Analysis(frame, boxes[keep], scores[keep], roi_px, size, crop.shape)


def analyze_photo(detector, image, s):
    """One still image. With "auto", a first pass at 640 measures the grains, then the best size is used."""
    if s["imgsz"] != "auto":
        return analyze(detector, image, s)
    a = analyze(detector, image, s, 640)
    best = recommend_imgsz(a.boxes, a.input_shape)
    return a if best in (None, 640) else analyze(detector, image, s, best)


def grain_size(a):
    """
    (grain_px, model_px, verdict) for the median grain, or None with fewer than 3 grains.
    grain_px: grain length in the camera picture. model_px: length inside the model after the
    region crop is resized to the input size. verdict: how well the camera height suits the model.
    """
    if a.count < 3:
        return None
    grain = float(np.median(np.maximum(a.boxes[:, 2] - a.boxes[:, 0], a.boxes[:, 3] - a.boxes[:, 1])))
    h0, w0 = a.input_shape[:2]
    h, w = best_shape(a.input_shape, a.imgsz)
    model = grain * min(h / h0, w / w0)
    # Too few real pixels can't be fixed by resizing; a big grain in a sharp picture can (it is shrunk).
    if grain < 25:
        verdict = "작음 - 카메라를 낮추세요"
    elif grain < 35:
        verdict = "조금 작음"
    elif model > 45:
        verdict = "모델에서 큼 - 입력 크기를 자동이나 더 작게"
    elif model < 18:
        verdict = "모델에서 작음 - 입력 크기를 자동이나 더 크게"
    else:
        verdict = "적당"
    return grain, model, verdict


class AutoSize:
    """Live video: follows recommend_imgsz, switching only after 3 frames agree (no flicker)."""

    def __init__(self, start=960):
        self.size, self._pending, self._votes = start, None, 0

    def update(self, a):
        best = recommend_imgsz(a.boxes, a.input_shape)
        if best is None or best == self.size:
            self._pending, self._votes = None, 0
            return
        self._votes = self._votes + 1 if best == self._pending else 1
        self._pending = best
        if self._votes >= 3:
            self.size, self._pending, self._votes = best, None, 0


# ---------------------------------------------------------------- drawing / image files
BOX_COLOR = (0, 90, 255)        # BGR orange-red, visible on brown grains and white paper
ROI_COLOR = (0, 200, 255)


def draw(a, caption=None):
    img = a.frame.copy()
    t = max(1, round(img.shape[1] / 900))
    for x1, y1, x2, y2 in a.boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, t)
    if a.roi_px is not None:
        x1, y1, x2, y2 = a.roi_px
        cv2.rectangle(img, (x1, y1), (x2, y2), ROI_COLOR, t + 1)
    if caption:
        scale = img.shape[1] / 1400
        org = (int(20 * scale), int(55 * scale))
        cv2.putText(img, caption, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale, (255, 255, 255),
                    int(7 * scale) + 1, cv2.LINE_AA)
        cv2.putText(img, caption, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale, (0, 0, 0),
                    int(3 * scale) + 1, cv2.LINE_AA)
    return img


def read_image(path):
    """cv2.imread that also works with Korean characters in the path."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(path))
    return ok


# ---------------------------------------------------------------- cameras
def frame_looks_garbled(frame):
    """Picture squeezed into the top rows with the rest black (requested size not supported)."""
    h = frame.shape[0]
    return frame[h // 2:].mean() < 5 and frame[: h // 4].mean() > 20


def _probe(index, backend):
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame.shape[1], frame.shape[0]
    finally:
        cap.release()


def list_cameras(max_index=10):
    """
    [(label, index, backend, wireless)] in DirectShow order, with device names.
    Windows' wireless "connected camera" is listed but not opened (opening it wakes the
    phone link); devices that fail under DirectShow (e.g. DroidCam) are retried via
    Media Foundation by name.
    """
    names = list_video_devices()
    msmf = list_msmf_video_devices()
    cams = []
    for i in range(max(max_index, len(names))):
        name = names[i] if i < len(names) else "카메라"
        if is_windows_virtual_camera(name):
            cams.append((f"{name} (무선)", i, cv2.CAP_DSHOW, True))
            continue
        index, backend = i, cv2.CAP_DSHOW
        size = _probe(index, backend)
        if size is None and i < len(names) and name in msmf:
            index, backend = msmf.index(name), cv2.CAP_MSMF
            size = _probe(index, backend)
        if size is not None:
            cams.append((f"{name} ({size[0]}x{size[1]})", index, backend, False))
    return cams


def preferred_camera(cams):
    """Index into cams: DroidCam first, never the wireless one, else the first usable."""
    usable = [k for k, c in enumerate(cams) if not c[3]]
    for k in usable:
        if "droidcam" in cams[k][0].lower():
            return k
    return usable[0] if usable else (0 if cams else None)


def open_camera(index, backend):
    """Open at 1920x1080 when the camera supports it, else at its default size."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if backend == cv2.CAP_MSMF:
        # Media Foundation virtual cameras may ignore the size and send a garbled frame.
        last = None
        for _ in range(10):
            ok, f = cap.read()
            if ok:
                last = f
        if last is None or frame_looks_garbled(last):
            cap.release()
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                return None
    return cap


class FpsMeter:
    def __init__(self, n=10):
        self.t = []
        self.n = n

    def tick(self):
        self.t.append(time.perf_counter())
        self.t = self.t[-self.n:]
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0
