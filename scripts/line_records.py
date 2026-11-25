"""
Per-line seed count log.

Each press of "Next" adds one batch (the current seed count, and optionally a saved photo) to a
plant line. A line's total is the sum of its batches. Every change is written straight to CSV
files (UTF-8 with BOM so Excel shows Korean names):

  seed_counts_<time>.csv          one row per batch
  seed_counts_<time>_summary.csv  one row per line
"""
import csv
from datetime import datetime
from pathlib import Path


class LineRecorder:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.out_dir / f"seed_counts_{self.stamp}.csv"
        self.summary_path = self.out_dir / f"seed_counts_{self.stamp}_summary.csv"
        self.image_dir = self.out_dir / f"images_{self.stamp}"
        self.records = []

    @classmethod
    def open_existing(cls, log_path):
        """
        Continue an earlier session: read its per-batch CSV (also files written by older versions
        of the program) and keep writing to the same files. Raises ValueError for other CSVs.
        """
        log_path = Path(log_path)
        rec = cls(log_path.parent)
        rec.log_path = log_path
        rec.summary_path = log_path.with_name(log_path.stem + "_summary.csv")
        stamp = log_path.stem[len("seed_counts_"):] if log_path.stem.startswith("seed_counts_") else log_path.stem
        rec.stamp = stamp
        rec.image_dir = log_path.parent / f"images_{stamp}"
        with open(log_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"line", "count"} <= set(reader.fieldnames):
                raise ValueError("라인 기록 CSV가 아닙니다 (line, count 열이 없음)")
            for row in reader:
                line = (row.get("line") or "").strip()
                if not line:
                    continue
                rec.records.append({
                    "time": row.get("time", ""),
                    "line": line,
                    "batch": rec.next_batch(line),
                    "count": int(float(row.get("count") or 0)),
                    "image": row.get("image", "") or "",
                })
        return rec

    def next_batch(self, line):
        return sum(1 for r in self.records if r["line"] == line) + 1

    def add(self, line, count, image=None):
        """Add one batch to `line`. Returns (batch_number, line_total)."""
        line = line.strip()
        if not line:
            raise ValueError("라인 이름이 비어 있습니다")
        self.records.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "line": line,
            "batch": self.next_batch(line),
            "count": int(count),
            "image": str(image) if image else "",
        })
        try:
            self.save()
        except OSError:
            # e.g. the CSV is open in Excel: keep memory and file in step, or a retry would count it twice
            self.records.pop()
            raise
        return self.records[-1]["batch"], self.total(line)

    def undo(self):
        """Remove the most recent batch. Returns it, or None. Raises OSError (and keeps it) if saving fails."""
        if not self.records:
            return None
        rec = self.records.pop()
        try:
            self.save()
        except OSError:
            self.records.append(rec)
            raise
        return rec

    def total(self, line):
        return sum(r["count"] for r in self.records if r["line"] == line)

    def batches(self, line):
        return sum(1 for r in self.records if r["line"] == line)

    def summary(self):
        """[(line, batches, total)] in first-counted order."""
        out = {}
        for r in self.records:
            b, t = out.get(r["line"], (0, 0))
            out[r["line"]] = (b + 1, t + r["count"])
        return [(line, b, t) for line, (b, t) in out.items()]

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        running = {}
        with open(self.log_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["time", "line", "batch", "count", "line_total", "image"])
            for r in self.records:
                running[r["line"]] = running.get(r["line"], 0) + r["count"]
                w.writerow([r["time"], r["line"], r["batch"], r["count"], running[r["line"]], r["image"]])
        with open(self.summary_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["line", "batches", "total"])
            w.writerows(self.summary())
