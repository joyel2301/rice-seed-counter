"""
볍씨 카운터 - live rice seed counting with a phone/USB camera or from photos.

Run:  python scripts/rice_counter.py
Check a packaged build without a camera:  RiceSeedCounter.exe --selftest photo.jpg out.txt
"""
import os
import re
import sys
import threading
from collections import Counter, deque

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPoint, QRect, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QRubberBand, QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import app_info
import app_log
import core
from line_records import LineRecorder

APP_NAME = app_info.APP_NAME
AUTO_SIZE = "자동 (권장)"
SIZES = [AUTO_SIZE, "640", "960", "1280"]

STYLE = """
QWidget { font-family: 'Malgun Gothic'; font-size: 10pt; color: #202124; }
QMainWindow, QWidget#root { background: #f4f5f7; }
QGroupBox { border: 1px solid #dadde2; border-radius: 8px; margin-top: 12px; padding: 12px 8px 8px 8px;
            background: #ffffff; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #3c4043; }
QPushButton { padding: 6px 12px; border: 1px solid #c8cdd4; border-radius: 6px; background: #ffffff; }
QPushButton:hover { background: #eef1f5; }
QPushButton:disabled { color: #a3a7ad; background: #f1f2f4; border-color: #e0e2e6; }
QPushButton#primary { background: #2e7d32; color: white; border: none; font-size: 13pt; font-weight: 700;
                      padding: 12px; }
QPushButton#primary:hover { background: #256b29; }
QPushButton#primary:disabled { background: #a5c9a7; }
QPushButton#start { background: #1a73e8; color: white; border: none; font-weight: 700; }
QPushButton#start:hover { background: #1765cc; }
QPushButton#start:disabled { background: #a9c7f3; }
QTabWidget::pane { border: 1px solid #dadde2; border-radius: 8px; background: #ffffff; top: -1px; }
QTabBar::tab { padding: 7px 14px; background: #eceef1; border: 1px solid #dadde2; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background: #ffffff; font-weight: 700; }
QLabel#count { font-size: 30pt; font-weight: 800; color: #1b5e20; }
QLabel#sub { color: #5f6368; font-size: 10.5pt; }
QLabel#lineTotal { font-size: 16pt; font-weight: 700; color: #1b5e20; }
QLabel#hint { color: #6b7076; font-size: 9pt; }
QTableWidget { border: 1px solid #e3e5e8; gridline-color: #eceef1; background: #ffffff; }
QHeaderView::section { background: #f1f3f5; border: none; border-bottom: 1px solid #dadde2; padding: 4px; }
"""


# ---------------------------------------------------------------- video view
class VideoView(QLabel):
    """Shows frames scaled to fit; lets the user drag a rectangle in frame coordinates."""
    selected = Signal(str, tuple)          # mode, (x1, y1, x2, y2) in frame pixels

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # never grow with the pixmap
        self.setStyleSheet("background: #1f2124; color: #a8adb3; border-radius: 8px; font-size: 12pt;")
        self._img = None
        self._mode = None
        self._origin = None
        self._band = QRubberBand(QRubberBand.Rectangle, self)

    def show_image(self, qimg):
        self._img = qimg
        self._rescale()

    def clear_image(self, text):
        self._img = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def has_image(self):
        return self._img is not None

    def _rescale(self):
        if self._img is None:
            return
        dpr = self.devicePixelRatioF()
        pm = QPixmap.fromImage(self._img).scaled(QSize(int(self.width() * dpr), int(self.height() * dpr)),
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    # --- rectangle selection
    def begin_select(self, mode):
        self._mode = mode
        self.setCursor(Qt.CrossCursor)

    def cancel_select(self):
        self._mode = None
        self._band.hide()
        self.unsetCursor()

    def selecting(self):
        return self._mode is not None

    def _to_frame(self, p):
        pm = self.pixmap()
        size = pm.deviceIndependentSize()
        ox, oy = (self.width() - size.width()) / 2, (self.height() - size.height()) / 2
        fx = (p.x() - ox) / size.width() * self._img.width()
        fy = (p.y() - oy) / size.height() * self._img.height()
        return int(np.clip(fx, 0, self._img.width())), int(np.clip(fy, 0, self._img.height()))

    def mousePressEvent(self, e):
        if self._mode and self._img is not None and e.button() == Qt.LeftButton:
            self._origin = e.position().toPoint()
            self._band.setGeometry(QRect(self._origin, QSize()))
            self._band.show()

    def mouseMoveEvent(self, e):
        if self._mode and self._origin is not None:
            self._band.setGeometry(QRect(self._origin, e.position().toPoint()).normalized())

    def mouseReleaseEvent(self, e):
        if not self._mode or self._origin is None:
            return
        x1, y1 = self._to_frame(self._origin)
        x2, y2 = self._to_frame(e.position().toPoint())
        mode = self._mode
        self._origin = None
        self.cancel_select()
        if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:
            self.selected.emit(mode, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))


# ---------------------------------------------------------------- background threads
class EngineLoader(QThread):
    done = Signal(object, str, str)        # detector, engine used, error

    def __init__(self, engine, imgsz):
        super().__init__()
        self.engine, self.imgsz = engine, imgsz

    def run(self):
        try:
            det, used = core.load_detector(self.engine, self.imgsz)
            self.done.emit(det, used, "")
        except Exception as e:
            app_log.log_exception(*sys.exc_info(), where=f"loading engine {self.engine}")
            self.done.emit(None, "", str(e))


class CameraWorker(QThread):
    """Reads the camera and runs detection; the GUI thread only displays results."""
    result = Signal(object, object, float, int)  # Analysis, display image (BGR), fps, settings version
    failed = Signal(str)

    def __init__(self, cap, detector, get_settings):
        super().__init__()
        self.cap, self.detector, self.get_settings = cap, detector, get_settings
        self.consumed = threading.Event()
        self.consumed.set()
        self.auto = core.AutoSize()      # input size chosen from the grain size ("자동")

    def run(self):
        fps = core.FpsMeter()
        while not self.isInterruptionRequested():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.failed.emit("카메라에서 영상을 읽지 못했습니다.")
                return
            s = self.get_settings()
            try:
                a = core.analyze(self.detector, frame, s, self.auto.size if s["imgsz"] == "auto" else None)
                if s["imgsz"] == "auto":
                    self.auto.update(a)
            except Exception as e:
                app_log.log_exception(*sys.exc_info(), where="live detection")
                self.failed.emit(f"검출 오류: {e}")
                return
            rate = fps.tick()
            if self.consumed.is_set():           # drop frames the GUI has no time to show
                self.consumed.clear()
                self.result.emit(a, core.draw(a), rate, s.get("ver", 0))


# ---------------------------------------------------------------- helpers
def to_qimage(bgr):
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()


def safe_filename(text):
    return re.sub(r'[\\/:*?"<>|]+', "_", text).strip() or "line"


def hint(text):
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        icon = core.bundle_path("assets/app_icon.ico")
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))

        self.prefs = QSettings("RiceSeedCounter", "RiceSeedCounter")
        self.recorder = LineRecorder(core.output_dir())
        self.engines = core.available_engines()
        self.detector = None
        self.engine_used = None
        self.loader = None
        self.photo_detector = None      # OpenVINO CPU detector for photo files
        self.cap = None
        self.worker = None
        self.cams = []
        self.photo = None
        self.photo_name = ""
        self.last = None                # latest Analysis
        self.recent = deque(maxlen=5)   # recent counts, so "다음" adds a steady value
        self._ver = 0                   # bumped on every settings change (stale frames are not counted)
        self._stopping = []             # (worker, cap) still finishing after a stop (camera read stuck)
        self.roi = self._load_roi()
        self._s = {}
        self._loading = True            # don't write preferences while restoring them

        self._build_ui()
        self._restore_prefs()
        self._loading = False
        self.update_settings()
        self.refresh_table()
        self.update_line_info()
        self.resize(1400, 860)
        QTimer.singleShot(50, self._startup)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 8)
        outer.setSpacing(12)

        # left: count + video
        left = QVBoxLayout()
        top = QHBoxLayout()
        self.count_label = QLabel("0알")
        self.count_label.setObjectName("count")
        self.count_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)   # never clip the number
        self.sub_label = QLabel("")
        self.sub_label.setObjectName("sub")
        top.addWidget(self.count_label)
        top.addSpacing(16)
        top.addWidget(self.sub_label, 1, Qt.AlignBottom)
        # grain size in the picture and inside the model, with a verdict on the camera height
        self.size_label = QLabel("")
        self.size_label.setToolTip("낟알 길이(중앙값): 카메라 화면에서 몇 픽셀인지 → 모델 안에서 몇 픽셀인지.\n"
                                   "모델은 낟알이 약 20~40픽셀일 때 가장 잘 셉니다. 카메라 화면에서는 35픽셀 이상이면 충분하고,\n"
                                   "더 크게 찍힌 건 입력 크기 자동이 줄여서 맞춥니다.")
        top.addWidget(self.size_label, 0, Qt.AlignBottom)
        left.addLayout(top)
        self.video = VideoView()
        self.video.clear_image("카메라를 시작하거나 사진을 여세요")
        self.video.selected.connect(self.on_rect_selected)
        left.addWidget(self.video, 1)
        outer.addLayout(left, 1)

        # right: panel
        panel = QWidget()
        panel.setFixedWidth(400)
        right = QVBoxLayout(panel)
        right.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

        cam_box = QGroupBox("카메라")
        cg = QVBoxLayout(cam_box)
        row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cam_combo.setMinimumContentsLength(18)
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        row.addWidget(self.cam_combo, 1)
        row.addWidget(self.refresh_btn)
        cg.addLayout(row)
        row = QHBoxLayout()
        self.start_btn = QPushButton("▶  시작")
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("■  정지")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.photo_btn = QPushButton("사진 열기…")
        self.photo_btn.clicked.connect(self.open_photo)
        row.addWidget(self.start_btn, 1)
        row.addWidget(self.stop_btn, 1)
        row.addWidget(self.photo_btn, 1)
        cg.addLayout(row)
        right.addWidget(cam_box)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_record_tab(), "라인 기록")
        self.tabs.addTab(self._build_detect_tab(), "검출 설정")
        self.tabs.addTab(self._build_region_tab(), "검출 영역")
        right.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        about_btn = QPushButton("정보")
        about_btn.clicked.connect(self.show_about)
        row.addWidget(about_btn)
        right.addLayout(row)

        self.statusBar().showMessage("준비 중…")
        self.set_running(False)

    def _build_record_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("라인 이름"))
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("예: IR64-1   (Enter = 다음)")
        self.line_edit.returnPressed.connect(self.add_batch)
        self.line_edit.textChanged.connect(self.update_line_info)
        v.addWidget(self.line_edit)
        self.line_total = QLabel("")
        self.line_total.setObjectName("lineTotal")
        v.addWidget(self.line_total)
        self.next_btn = QPushButton("다음  —  현재 개수 추가")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self.add_batch)
        v.addWidget(self.next_btn)
        row = QHBoxLayout()
        self.undo_btn = QPushButton("마지막 취소")
        self.undo_btn.clicked.connect(self.undo_batch)
        self.save_photo_cb = QCheckBox("다음 누를 때 사진 저장")
        self.save_photo_cb.setChecked(True)
        row.addWidget(self.undo_btn)
        row.addStretch(1)
        row.addWidget(self.save_photo_cb)
        v.addLayout(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["라인", "횟수", "합계"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 3):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.cellClicked.connect(lambda r, _c: self.line_edit.setText(self.table.item(r, 0).text()))
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.path_label = hint(f"저장: {self.recorder.log_path.name}")
        folder_btn = QPushButton("폴더 열기")
        folder_btn.clicked.connect(self.open_folder)
        resume_btn = QPushButton("이전 기록 이어하기…")
        resume_btn.setToolTip("results 폴더의 이전 기록 CSV를 불러와서, 같은 파일에 이어서 기록합니다.")
        resume_btn.clicked.connect(self.resume_records)
        row.addWidget(self.path_label, 1)
        row.addWidget(resume_btn)
        row.addWidget(folder_btn)
        v.addLayout(row)
        return w

    def resume_records(self):
        d = core.output_dir()
        d.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, "이어서 기록할 CSV 선택", str(d),
                                              "라인 기록 CSV (seed_counts_*.csv);;CSV (*.csv)")
        if not path:
            return
        if path.endswith("_summary.csv"):
            path = path[: -len("_summary.csv")] + ".csv"     # the per-batch file holds the data
        try:
            rec = LineRecorder.open_existing(path)
        except (OSError, ValueError) as e:
            app_log.log(f"resume failed for {path}: {e!r}")
            QMessageBox.warning(self, APP_NAME, f"기록을 불러오지 못했습니다:\n{path}\n\n{e}")
            return
        if self.recorder.records and Path(path) != self.recorder.log_path:
            if QMessageBox.question(
                    self, APP_NAME,
                    f"지금까지 기록한 {len(self.recorder.records)}회는 {self.recorder.log_path.name}에 저장돼 있습니다.\n"
                    f"선택한 기록({Path(path).name})으로 바꿔서 이어갈까요?") != QMessageBox.Yes:
                return
        self.recorder = rec
        self.path_label.setText(f"저장: {rec.log_path.name}")
        self.refresh_table()
        self.update_line_info()
        lines = len(rec.summary())
        self.statusBar().showMessage(f"이어하기: {rec.log_path.name} — 라인 {lines}개, {len(rec.records)}회 불러옴")

    def _build_detect_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(self.engines)
        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        f.addRow("엔진", self.engine_combo)
        self.size_combo = QComboBox()
        self.size_combo.addItems(SIZES)
        self.size_combo.currentTextChanged.connect(self.update_settings)
        f.addRow("입력 크기", self.size_combo)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.50)
        self.conf_spin.valueChanged.connect(self.update_settings)
        f.addRow("신뢰도", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.10, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.50)
        self.iou_spin.valueChanged.connect(self.update_settings)
        f.addRow("IoU", self.iou_spin)
        self.filter_cb = QCheckBox("오검출 필터 (무채색 물체, 크기가 튀는 박스 제외)")
        self.filter_cb.toggled.connect(self.update_settings)
        f.addRow("", self.filter_cb)
        f.addRow(hint(
            "• 엔진: 인텔 그래픽이 있으면 GPU FP16이 가장 빠릅니다. 엔진 변경은 카메라 정지 후 가능합니다.\n"
            "• 입력 크기: 자동(권장)이면 낟알이 찍힌 크기를 보고 맞는 값을 고릅니다. 너무 크게 넣으면 "
            "한 낟알이 두 개로 잡히고, 너무 작게 넣으면 놓칩니다.\n"
            "• 신뢰도: 높이면 애매한 검출이 줄고, 낮추면 놓치는 낟알이 줄어듭니다.\n"
            "• IoU: 이 값보다 많이 겹친 박스는 하나로 합칩니다. 너무 낮으면 붙어 있는 낟알을 하나로 지우고, "
            "너무 높으면 한 낟알이 두 개로 잡힙니다. 보통 0.5를 쓰세요.\n"
            "• 사진 파일은 OpenVINO CPU(FP32)로 분석합니다. 가로·세로·정사각형 사진 모두 해상도 손해 없이 처리합니다."))
        return w

    def _build_region_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        roi_box = QGroupBox("검출 영역")
        rv = QVBoxLayout(roi_box)
        self.roi_label = QLabel("")
        rv.addWidget(self.roi_label)
        row = QHBoxLayout()
        b = QPushButton("영역 지정")
        b.clicked.connect(lambda: self.begin_select("roi", "영상 위에서 셀 영역을 드래그하세요 (Esc 취소)"))
        row.addWidget(b)
        b = QPushButton("전체 화면")
        b.clicked.connect(self.clear_roi)
        row.addWidget(b)
        rv.addLayout(row)
        rv.addWidget(hint("지정한 영역만 잘라서 모델에 넣고, 영역 안에 중심이 있는 낟알만 셉니다. "
                          "빈 바닥이나 판 옆면을 빼면 모델 해상도가 낟알에 집중돼서 더 정확하고 빨라집니다. "
                          "동전 같은 물체를 빼고 셀 때도 사용합니다."))
        v.addWidget(roi_box)

        v.addStretch(1)
        return w

    # ------------------------------------------------------------ preferences
    def _load_roi(self):
        roi = self.prefs.value("roi", "")
        try:
            vals = [float(x) for x in str(roi).split(",")] if roi else []
            return tuple(vals) if len(vals) == 4 else None
        except ValueError:
            return None

    def _restore_prefs(self):
        p = self.prefs
        eng = p.value("engine", "")
        if eng in self.engines:
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText(eng)
            self.engine_combo.blockSignals(False)
        self.size_combo.setCurrentText(p.value("input_size", AUTO_SIZE))
        self.conf_spin.setValue(float(p.value("conf", 0.50)))
        self.iou_spin.setValue(float(p.value("iou", 0.50)))
        self.filter_cb.setChecked(p.value("filter", "false") == "true")
        self.save_photo_cb.setChecked(p.value("save_photo", "true") == "true")

    def _save_prefs(self):
        if self._loading:
            return
        p = self.prefs
        p.setValue("engine", self.engine_combo.currentText())
        p.setValue("input_size", self.size_combo.currentText())
        p.setValue("conf", self.conf_spin.value())
        p.setValue("iou", self.iou_spin.value())
        p.setValue("filter", "true" if self.filter_cb.isChecked() else "false")
        p.setValue("save_photo", "true" if self.save_photo_cb.isChecked() else "false")
        p.setValue("roi", ",".join(f"{x:.5f}" for x in self.roi) if self.roi else "")

    # ------------------------------------------------------------ startup / engine
    def _startup(self):
        self.load_engine()
        self.refresh_cameras()

    def load_engine(self):
        if self.loader is not None:
            return
        self.detector = None
        self.engine_combo.setEnabled(False)
        self.set_running(False)
        self.statusBar().showMessage("엔진 준비 중… (처음 GPU를 쓸 때는 20초 정도 걸릴 수 있습니다)")
        self.loader = EngineLoader(self.engine_combo.currentText(), self.fixed_size() or 960)
        self.loader.done.connect(self.on_engine_loaded)
        self.loader.start()

    def on_engine_loaded(self, det, used, err):
        self.loader.wait()
        self.loader = None
        self.engine_combo.setEnabled(True)
        if det is None:
            QMessageBox.critical(self, APP_NAME, f"모델을 불러오지 못했습니다.\n\n{err}")
            self.statusBar().showMessage("모델을 불러오지 못했습니다.")
            return
        self.detector, self.engine_used = det, used
        wanted = self.engine_combo.currentText()
        if used != wanted:
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText(used)
            self.engine_combo.blockSignals(False)
            self.statusBar().showMessage(f"{wanted}을(를) 쓸 수 없어 {used}(으)로 전환했습니다.")
        else:
            self.statusBar().showMessage(f"준비 완료 — 엔진: {used}")
        self._save_prefs()
        self.set_running(False)

    def on_engine_changed(self, _text):
        if self.worker is None:
            self.load_engine()

    # ------------------------------------------------------------ settings
    def update_settings(self, *_):
        self._s = {
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "imgsz": self.fixed_size() or "auto",
            "filter": self.filter_cb.isChecked(),
            "roi": self.roi,
            "ver": self._ver + 1,
        }
        self._ver += 1
        self.recent.clear()     # counts from the old settings must not reach "다음"
        if self.roi:
            x, y, w, h = self.roi
            self.roi_label.setText(f"영역: 가로 {x*100:.0f}~{(x+w)*100:.0f}%, 세로 {y*100:.0f}~{(y+h)*100:.0f}%")
        else:
            self.roi_label.setText("영역: 전체 화면")
        self._save_prefs()
        if self.worker is None and self.photo is not None:
            self.analyze_photo()

    def settings(self):
        return self._s

    def fixed_size(self):
        """Chosen input size as int, or None for automatic."""
        t = self.size_combo.currentText()
        return None if t == AUTO_SIZE else int(t)

    # ------------------------------------------------------------ cameras
    def refresh_cameras(self):
        if self.worker is not None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("카메라 찾는 중…")
        QApplication.processEvents()
        try:
            self.cams = core.list_cameras()
        finally:
            QApplication.restoreOverrideCursor()
        self.cam_combo.clear()
        self.cam_combo.addItems([c[0] for c in self.cams])
        last = self.prefs.value("camera", "")
        names = [c[0].rsplit(" (", 1)[0] for c in self.cams]
        k = names.index(last) if last in names and not self.cams[names.index(last)][3] else core.preferred_camera(self.cams)
        if k is not None:
            self.cam_combo.setCurrentIndex(k)
        if self.loader is None:
            self.statusBar().showMessage(f"카메라 {len(self.cams)}개 찾음" if self.cams else "카메라가 없습니다.")
        self.set_running(False)

    def start_camera(self):
        if self.worker is not None or self.detector is None:
            return
        if self._stopping:
            QMessageBox.information(self, APP_NAME, "이전 카메라 연결을 정리하는 중입니다. 잠시 후 다시 시작하세요.")
            return
        k = self.cam_combo.currentIndex()
        if k < 0 or k >= len(self.cams):
            QMessageBox.warning(self, APP_NAME, "카메라를 선택하세요.")
            return
        label, index, backend, wireless = self.cams[k]
        if wireless and QMessageBox.question(
                self, APP_NAME, "Windows 무선 카메라는 휴대폰 연결(Windows와 연결)을 깨웁니다.\n계속할까요?"
        ) != QMessageBox.Yes:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cap = core.open_camera(index, backend)
        finally:
            QApplication.restoreOverrideCursor()
        if cap is None:
            QMessageBox.critical(self, APP_NAME, f"카메라를 열 수 없습니다:\n{label}")
            return
        self.prefs.setValue("camera", label.rsplit(" (", 1)[0])
        self.cap = cap
        self.photo = None
        self.recent.clear()
        self.worker = CameraWorker(cap, self.detector, self.settings)
        self.worker.result.connect(self.on_result)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()
        self.set_running(True)
        self.statusBar().showMessage(f"실행 중 — {label} · {self.engine_used}")

    def stop_camera(self, message="정지했습니다."):
        worker, cap = self.worker, self.cap
        self.worker = self.cap = None
        if worker is not None:
            worker.requestInterruption()
            if not worker.wait(3000):
                # The camera read is stuck (e.g. DroidCam disconnected). Never destroy a running
                # thread or release the camera under it: let it finish, then release the camera.
                self._stopping.append((worker, cap))
                worker.finished.connect(lambda w=worker: self._finish_stop(w))
                cap = None
                message += " (카메라 연결을 정리하는 중)"
        if cap is not None:
            cap.release()
        self.set_running(False)
        self.statusBar().showMessage(message)

    def _finish_stop(self, worker):
        for item in list(self._stopping):
            if item[0] is worker:
                if item[1] is not None:
                    item[1].release()
                self._stopping.remove(item)

    def on_worker_failed(self, msg):
        if self.sender() is not self.worker:    # a camera the user already stopped: nothing to report
            return
        self.stop_camera(msg)
        QMessageBox.warning(self, APP_NAME, msg)

    def on_result(self, a, disp, fps, ver):
        if self.worker is None:
            return
        # frames analysed with settings older than the current ones are shown but not counted
        self.show_analysis(a, disp, fps, current=(ver == self._ver))
        self.worker.consumed.set()

    def set_running(self, running):
        idle = not running
        self.start_btn.setEnabled(idle and self.detector is not None and bool(self.cams))
        self.stop_btn.setEnabled(running)
        self.cam_combo.setEnabled(idle)
        self.refresh_btn.setEnabled(idle)
        self.engine_combo.setEnabled(idle and self.loader is None)

    # ------------------------------------------------------------ photos
    def open_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "사진 열기", self.prefs.value("photo_dir", ""),
                                              "이미지 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path:
            return
        img = core.read_image(path)
        if img is None:
            QMessageBox.warning(self, APP_NAME, f"사진을 열 수 없습니다:\n{path}")
            return
        if self.worker is not None:
            self.stop_camera()
        self.prefs.setValue("photo_dir", os.path.dirname(path))
        self.photo, self.photo_name = img, os.path.basename(path)
        self.analyze_photo()

    def analyze_photo(self):
        if self.photo is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("사진 분석 중…")
        QApplication.processEvents()
        try:
            if self.photo_detector is None:
                self.photo_detector = core.load_photo_detector()
            a = core.analyze_photo(self.photo_detector, self.photo, self._s)
        except Exception as e:
            app_log.log_exception(*sys.exc_info(), where=f"photo analysis {self.photo_name}")
            QMessageBox.critical(self, APP_NAME, f"사진 분석 오류:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.recent.clear()
        self.show_analysis(a, core.draw(a), None)
        h, w = self.photo.shape[:2]
        self.statusBar().showMessage(f"사진: {self.photo_name} ({w}x{h}) · OpenVINO CPU · 입력 {a.imgsz}")

    # ------------------------------------------------------------ display
    def show_analysis(self, a, disp, fps, current=True):
        if current:
            self.last = a
            self.recent.append(a.count)
        self.video.show_image(to_qimage(disp))
        self.count_label.setText(f"{a.count:,}알")
        parts = [f"입력 {a.imgsz}" + (" (자동)" if self._s.get("imgsz") == "auto" else "")]
        if fps is not None:
            parts.append(f"안정값 {self.stable_count():,}")
            parts.append(f"{fps:.1f} fps")
        self.sub_label.setText("   ·   ".join(parts))
        self.show_grain_size(a)

    SIZE_COLORS = {"적당": ("#e6f4ea", "#1e6b30"), "조금 작음": ("#fef3e0", "#8a5300")}

    def show_grain_size(self, a):
        info = core.grain_size(a)
        if info is None:
            self.size_label.setText("")
            self.size_label.setStyleSheet("")
            return
        grain, model, verdict = info
        bg, fg = self.SIZE_COLORS.get(verdict, ("#fde8e8", "#a61b1b") if verdict.startswith("작음")
                                      else ("#fef3e0", "#8a5300"))
        self.size_label.setText(f"낟알 {grain:.0f}px → 모델 {model:.0f}px · {verdict}")
        self.size_label.setStyleSheet(f"background: {bg}; color: {fg}; border-radius: 10px; "
                                      f"padding: 3px 10px; font-weight: 600;")

    def stable_count(self):
        """Most frequent of the recent counts (ties: the newest)."""
        if not self.recent:
            return 0
        freq = Counter(self.recent)
        best = max(freq.values())
        return next(c for c in reversed(self.recent) if freq[c] == best)

    # ------------------------------------------------------------ region
    def begin_select(self, mode, msg):
        if not self.video.has_image():
            QMessageBox.information(self, APP_NAME, "먼저 카메라를 시작하거나 사진을 여세요.")
            return
        self.video.begin_select(mode)
        self.statusBar().showMessage(msg)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.video.selecting():
            self.video.cancel_select()
            self.statusBar().showMessage("취소했습니다.")
            return
        super().keyPressEvent(e)

    def on_rect_selected(self, mode, rect):
        if self.last is None:
            return
        h, w = self.last.frame.shape[:2]
        x1, y1, x2, y2 = rect
        if mode == "roi":
            self.roi = (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)
            self.update_settings()
            self.statusBar().showMessage("검출 영역을 지정했습니다.")

    def clear_roi(self):
        self.roi = None
        self.update_settings()

    # ------------------------------------------------------------ line record
    def add_batch(self):
        name = self.line_edit.text().strip()
        if not name:
            QMessageBox.information(self, APP_NAME, "라인 이름을 입력하세요.")
            self.line_edit.setFocus()
            return
        if self.last is None:
            QMessageBox.information(self, APP_NAME, "아직 검출 결과가 없습니다. 카메라를 시작하거나 사진을 여세요.")
            return
        if self.worker is not None and not self.recent:
            self.statusBar().showMessage("바뀐 설정을 반영하는 중입니다. 잠시 후 다시 [다음]을 누르세요.")
            return
        a = self.last
        count = self.stable_count() if self.worker is not None else a.count
        image = None
        if self.save_photo_cb.isChecked():
            batch = self.recorder.next_batch(name)
            image = self.recorder.image_dir / f"{safe_filename(name)}_{batch:02d}_{count}알.jpg"
            if not self.save_photo(a, image, f"{name}  ·  {batch}회차  ·  {count:,}알"):
                image = None
        try:
            batch, total = self.recorder.add(name, count, image)
        except OSError as e:
            if image:                      # the batch was not recorded, so drop its photo too
                try:
                    os.remove(image)
                except OSError:
                    pass
            QMessageBox.critical(self, APP_NAME, "CSV를 저장하지 못해서 이번 회차는 기록하지 않았습니다.\n"
                                 f"엑셀에서 CSV 파일을 닫고 다시 [다음]을 누르세요.\n\n{e}")
            return
        self.refresh_table()
        self.update_line_info()
        self.statusBar().showMessage(f"'{name}' {batch}회차 +{count:,}알 → 합계 {total:,}알")

    def save_photo(self, a, path, caption):
        img = to_qimage(core.draw(a))
        p = QPainter(img)
        size = max(14, img.width() // 45)
        font = QFont("Malgun Gothic", 1)
        font.setPixelSize(size)
        font.setBold(True)
        p.setFont(font)
        rect = p.fontMetrics().boundingRect(caption).adjusted(-size // 2, -size // 4, size // 2, size // 4)
        rect.moveTopLeft(QPoint(size // 2, size // 2))
        p.fillRect(rect, QColor(255, 255, 255, 215))
        p.setPen(QPen(QColor(20, 20, 20)))
        p.drawText(rect, Qt.AlignCenter, caption)
        p.end()
        path.parent.mkdir(parents=True, exist_ok=True)
        return img.save(str(path), "JPG", 92)

    def undo_batch(self):
        try:
            rec = self.recorder.undo()
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, "CSV를 저장하지 못해서 취소하지 않았습니다.\n"
                                 f"엑셀에서 CSV 파일을 닫고 다시 시도하세요.\n\n{e}")
            return
        if rec is None:
            self.statusBar().showMessage("취소할 기록이 없습니다.")
            return
        if rec["image"]:
            try:
                os.remove(rec["image"])
            except OSError:
                pass
        self.refresh_table()
        self.update_line_info()
        self.statusBar().showMessage(f"취소: '{rec['line']}' {rec['batch']}회차 ({rec['count']:,}알)")

    def refresh_table(self):
        rows = self.recorder.summary()
        self.table.setRowCount(len(rows))
        for r, (line, batches, total) in enumerate(rows):
            vals = [line, str(batches), f"{total:,}"]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self.undo_btn.setEnabled(bool(self.recorder.records))

    def update_line_info(self, *_):
        name = self.line_edit.text().strip()
        if not name:
            self.line_total.setText("라인 이름을 입력하세요")
            return
        self.line_total.setText(f"합계 {self.recorder.total(name):,}알  ·  {self.recorder.batches(name)}회")

    def show_about(self):
        a = app_info
        box = QMessageBox(self)
        box.setWindowTitle(f"{a.APP_NAME} 정보")
        box.setTextFormat(Qt.RichText)
        box.setText(f"<h3>{a.APP_NAME} ({a.APP_NAME_EN}) {a.VERSION}</h3>"
                    f"<p><b>만든 사람</b>: {a.AUTHOR}, {a.AFFILIATION}<br>"
                    f"© {a.YEAR} {a.AUTHOR}. 프로그램 코드는 MIT License로 배포됩니다.<br>"
                    f"<span style='color:#6b7076; font-size:9pt'>{a.AI_CREDIT}</span></p>")
        box.setInformativeText(
            "<p><b>쌀알 검출 모델</b><br>BearHuntingExpert, Rice-Grain-Phenotyping-System (MIT License)<br>"
            "<a href='https://github.com/BearHuntingExpert/Rice-Grain-Phenotyping-System'>"
            "github.com/BearHuntingExpert/Rice-Grain-Phenotyping-System</a></p>"
            "<p><b>사용한 오픈소스</b><br>OpenVINO (Apache-2.0), OpenCV (Apache-2.0), NumPy (BSD), "
            "Qt / PySide6 (LGPL-3.0), Python (PSF)</p>"
            "<p>각 라이선스 전문과 고지는 licenses 폴더에 있습니다.</p>")
        lic_btn = box.addButton("라이선스 폴더 열기", QMessageBox.ActionRole)
        box.addButton("닫기", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is lic_btn:
            d = core.licenses_dir()
            if d.exists():
                os.startfile(str(d))
            else:
                QMessageBox.information(self, APP_NAME, f"licenses 폴더를 찾을 수 없습니다:\n{d}")

    def open_folder(self):
        d = core.output_dir()
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))

    def closeEvent(self, e):
        self.stop_camera()
        for worker, _cap in list(self._stopping):     # give a stuck camera read a few seconds to end
            worker.wait(5000)
        if self.loader is not None:
            self.loader.wait(30000)
        self._save_prefs()
        super().closeEvent(e)


# ---------------------------------------------------------------- uncaught errors
class ErrorReporter(QObject):
    """Shows uncaught errors to the user (from any thread) after app_log has written them down."""
    report = Signal(str)

    def __init__(self, parent_window):
        super().__init__()
        self.win = parent_window
        self._last = 0.0
        self.report.connect(self._show)          # queued to the GUI thread when emitted elsewhere

    def _show(self, text):
        import time
        if time.time() - self._last < 5:         # one dialog for a burst of errors
            return
        self._last = time.time()
        last_line = text.strip().splitlines()[-1] if text.strip() else ""
        QMessageBox.warning(self.win, APP_NAME,
                            "예상하지 못한 오류가 발생했습니다.\n"
                            f"자세한 내용은 다음 파일에 저장되었습니다:\n{app_log.path()}\n"
                            "문제가 계속되면 이 파일을 만든 사람에게 보내 주세요.\n\n" + last_line)


# ---------------------------------------------------------------- self-test (no GUI)
def self_test(image_path, out_path):
    img = core.read_image(image_path)
    s = dict(conf=0.5, iou=0.5, imgsz="auto", filter=False, roi=None)
    lines = []
    for engine in core.available_engines():
        try:
            det, used = core.load_detector(engine)
            a = core.analyze_photo(det, img, s)
            lines.append(f"{used}: {a.count} (input {a.imgsz})")
        except Exception as e:
            app_log.log_exception(*sys.exc_info(), where=f"self-test {engine}")
            lines.append(f"{engine}: ERROR {e}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    app_log.set_path(core.output_dir() / "error_log.txt")
    app_log.install()                       # log errors even before the window exists
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest":
        self_test(sys.argv[2], sys.argv[3])
        return
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in [(QPalette.Window, "#f4f5f7"), (QPalette.Base, "#ffffff"),
                        (QPalette.AlternateBase, "#f7f8fa"), (QPalette.Text, "#202124"),
                        (QPalette.WindowText, "#202124"), (QPalette.Button, "#ffffff"),
                        (QPalette.ButtonText, "#202124"), (QPalette.Highlight, "#1a73e8"),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#ffffff"),
                        (QPalette.ToolTipText, "#202124"), (QPalette.PlaceholderText, "#9aa0a6")]:
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    reporter = ErrorReporter(win)
    app_log.install(reporter.report.emit)   # from now on also tell the user
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
