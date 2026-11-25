"""
OpenVINO runner for the rice-grain model. Uses only OpenVINO, OpenCV and NumPy (no PyTorch).

The model was converted once per input shape (scripts/convert_openvino.py) into
external_models/rice_grain_phenotyping/openvino/best_<H>x<W>.xml. For each input-size option
there is a landscape (16:9), square and portrait (9:16) shape; each picture is fed through the
shape that keeps the most resolution.

Pre- and post-processing reproduce Ultralytics 8.3 exactly (LetterBox with auto=False, single-class
non_max_suppression, scale_boxes), so the boxes match what Ultralytics + PyTorch produce.

Measured on this PC (i5-1340P + Iris Xe, 1280 input): PyTorch CPU 477 ms/frame,
OpenVINO CPU 188 ms, OpenVINO GPU (FP16) 48 ms.
"""
from pathlib import Path

import cv2
import numpy as np

# Input-size option -> shapes (height, width): landscape 16:9, square, portrait 9:16
OV_SHAPES = {
    1280: [(736, 1280), (1280, 1280), (1280, 736)],
    960: [(544, 960), (960, 960), (960, 544)],
    640: [(384, 640), (640, 640), (640, 384)],
}
MAX_NMS = 30000          # as Ultralytics: at most this many candidates go into NMS


def openvino_available(ov_dir):
    try:
        import openvino  # noqa: F401
    except ImportError:
        return False
    return all((Path(ov_dir) / f"best_{h}x{w}.xml").exists() for h, w in landscape_shapes())


def landscape_shapes():
    return [group[0] for group in OV_SHAPES.values()]


def openvino_devices():
    try:
        import openvino as ov
        return ov.Core().available_devices
    except Exception:
        return []


# ---------------------------------------------------------------- Ultralytics-equivalent steps
def letterbox(img, new_shape):
    """Resize keeping aspect ratio and pad with grey (114) to new_shape (h, w), centred."""
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


class _OVNms:
    """
    Greedy NMS run by OpenVINO's NonMaxSuppression op on the CPU: same rule and order as
    torchvision.ops.nms (highest score first, drop boxes with IoU > iou_thres). ~1.5 ms for 11,000
    candidates, where NumPy or OpenCV need ~70 ms.
    """
    def __init__(self, core):
        import openvino as ov
        import openvino.opset13 as ops
        boxes = ops.parameter([1, -1, 4], np.float32)
        scores = ops.parameter([1, 1, -1], np.float32)
        max_out = ops.parameter([], np.int64)
        iou = ops.parameter([], np.float32)
        node = ops.non_max_suppression(boxes, scores, max_out, iou, ops.constant(np.float32(0.0)),
                                       box_encoding="corner", sort_result_descending=True,
                                       output_type="i64")
        model = ov.Model([node.output(0)], [boxes, scores, max_out, iou])
        self.req = core.compile_model(model, "CPU").create_infer_request()

    def __call__(self, boxes, scores, iou_thres, max_det):
        if len(boxes) == 0:
            return np.zeros(0, dtype=np.int64)
        out = self.req.infer({0: np.ascontiguousarray(boxes[None], dtype=np.float32),
                              1: np.ascontiguousarray(scores[None, None], dtype=np.float32),
                              2: np.int64(max_det), 3: np.float32(iou_thres)})[0]
        keep = out[:, 2]
        return keep[keep >= 0]


def nms_numpy(boxes, scores, iou_thres):
    """Reference greedy NMS in NumPy (slow; for checking _OVNms)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


def postprocess(raw, conf, iou, max_det, model_shape, image_shape, nms):
    """raw: (1, 4 + 1, N) model output -> (boxes Nx4 xyxy in image pixels, scores N)."""
    pred = raw[0].T                                    # (N, 5): cx, cy, w, h, score
    scores = pred[:, 4]
    pred = pred[scores > conf]
    if not len(pred):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    if len(pred) > MAX_NMS:
        pred = pred[np.argsort(-pred[:, 4], kind="stable")[:MAX_NMS]]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2] / 2, pred[:, 3] / 2
    boxes = np.stack([cx - w, cy - h, cx + w, cy + h], 1)
    scores = pred[:, 4]
    keep = nms(boxes, scores, iou, max_det)[:max_det]
    boxes, scores = boxes[keep], scores[keep]
    # back to the original picture (scale_boxes + clip_boxes)
    gain = min(model_shape[0] / image_shape[0], model_shape[1] / image_shape[1])
    pad_x = round((model_shape[1] - image_shape[1] * gain) / 2 - 0.1)
    pad_y = round((model_shape[0] - image_shape[0] * gain) / 2 - 0.1)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])
    return boxes.astype(np.float32), scores.astype(np.float32)


def best_shape(image_shape, imgsz):
    """Model shape for this picture: the one that scales it up most; ties -> fewer pixels."""
    h0, w0 = image_shape[:2]
    shapes = OV_SHAPES.get(imgsz, OV_SHAPES[1280])
    return max(shapes, key=lambda s: (round(min(s[0] / h0, s[1] / w0), 6), -s[0] * s[1]))


# ---------------------------------------------------------------- detector
class OVDetector:
    def __init__(self, ov_dir, device="GPU", precision="f16"):
        """precision applies to GPU only: "f16" (fastest; a grain right at the confidence
        line can occasionally flip) or "f32" (matches PyTorch, ~35% slower). CPU is always f32."""
        import openvino as ov
        self.ov_dir = Path(ov_dir)
        self.device = device
        self.precision = precision
        self.core = ov.Core()
        # Cache compiled GPU kernels on disk: first GPU compile takes ~20 s, later ones ~1-2 s.
        self.core.set_property({"CACHE_DIR": str(self.ov_dir / "cache")})
        self._compiled = {}  # (h, w) -> infer request
        self._nms = _OVNms(self.core)

    def _request(self, shape):
        if shape not in self._compiled:
            # Keep only the shape in use: each compiled shape holds its own working memory
            # (square 1280 on the GPU ~1.1 GB), and switching back costs only ~1-2 s thanks to CACHE_DIR.
            self._compiled.clear()
            import gc
            gc.collect()
            from openvino import Layout, Type
            from openvino.preprocess import ColorFormat, PrePostProcessor
            h, w = shape
            model = self.core.read_model(str(self.ov_dir / f"best_{h}x{w}.xml"))
            # Let the device do the uint8 -> float, BGR -> RGB, /255 and HWC -> CHW steps.
            ppp = PrePostProcessor(model)
            ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")) \
                .set_color_format(ColorFormat.BGR)
            ppp.input().model().set_layout(Layout("NCHW"))
            ppp.input().preprocess().convert_element_type(Type.f32) \
                .convert_color(ColorFormat.RGB).scale(255.0)
            model = ppp.build()
            config = {"PERFORMANCE_HINT": "LATENCY"}
            if self.device == "GPU":
                config["INFERENCE_PRECISION_HINT"] = self.precision
                # Default throttle keeps one CPU core spinning while the GPU works.
                config["GPU_QUEUE_THROTTLE"] = "LOW"
            self._compiled[shape] = self.core.compile_model(model, self.device, config).create_infer_request()
        return self._compiled[shape]

    def warm_up(self, imgsz=1280, image_shape=(1080, 1920)):
        shape = best_shape(image_shape, imgsz)
        self._request(shape).infer({0: np.zeros((1, *shape, 3), dtype=np.uint8)})

    def detect(self, image, conf=0.5, iou=0.5, imgsz=1280, max_det=3000):
        """image: BGR uint8. Returns (boxes Nx4 xyxy float32, scores N)."""
        shape = best_shape(image.shape, int(imgsz))
        x = np.ascontiguousarray(letterbox(image, shape))[None]
        raw = self._request(shape).infer({0: x})[0]
        return postprocess(raw, conf, iou, max_det, shape, image.shape, self._nms)
