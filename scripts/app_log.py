"""
Error log for the packaged app (it has no console): results/error_log.txt.

log(message)            one line with a timestamp (e.g. "GPU failed, using CPU")
log_exception(...)      a full traceback
install(on_error)       catch every uncaught exception (main thread, Python threads, Qt slots),
                        write it to the log and call on_error(text) so the GUI can tell the user
"""
import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

_path = None
_lock = threading.Lock()
_header_written = False


def set_path(path):
    global _path
    _path = Path(path)


def path():
    return _path


def _write(text):
    global _header_written
    if _path is None:
        return
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(_path, "a", encoding="utf-8") as f:
            if not _header_written:
                import app_info
                f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S}  {app_info.APP_NAME_EN} {app_info.VERSION}  "
                        f"Windows {platform.version()}  Python {platform.python_version()} =====\n")
                _header_written = True
            f.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        pass        # never let logging itself crash the app


def log(message):
    _write(f"[{datetime.now():%H:%M:%S}] {message}")


def log_exception(exc_type, exc, tb, where=""):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    _write(f"[{datetime.now():%H:%M:%S}] ERROR {where}\n{text}")
    return text


def install(on_error=None):
    """Route uncaught exceptions to the log (and on_error) instead of losing them."""
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = log_exception(exc_type, exc, tb, "(uncaught)")
        if sys.stderr is not None:
            sys.__excepthook__(exc_type, exc, tb)
        if on_error is not None:
            try:
                on_error(text)
            except Exception:
                pass

    sys.excepthook = hook
    threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)
