"""
file_watcher.py - Background watcher for incoming emergency reports.

Uses watchdog to detect when already-uploaded files are modified on disk.
Changes are put in a thread-safe queue; the Streamlit UI drains the queue
on each render and re-indexes changed files automatically.
"""

import queue
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import EMERGENCY_REPORT_DIR, EMERGENCY_REPORT_EXTENSIONS

SUPPORTED_EXTENSIONS = set(EMERGENCY_REPORT_EXTENSIONS)


def is_emergency_report(path: str | Path) -> bool:
    """Accept only supported files directly inside the report directory.

    Resolving paths also excludes symlinks pointing outside that directory.
    """
    candidate = Path(path).resolve()
    return (
        candidate.parent == Path(EMERGENCY_REPORT_DIR).resolve()
        and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        and candidate.is_file()
    )

# Module-level queue — shared across all Streamlit sessions in the process
_change_queue: queue.Queue = queue.Queue()
_observer_lock = threading.Lock()
_observer: Observer | None = None


class _UploadHandler(FileSystemEventHandler):
    """Puts (event_type, file_path) tuples into the queue on file changes."""

    def _handle(self, event, event_type: str):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_emergency_report(path):
            _change_queue.put((event_type, str(path)))

    def on_modified(self, event):
        self._handle(event, "modified")

    def on_created(self, event):
        self._handle(event, "created")


def start_watcher() -> None:
    """
    Start the watchdog observer for emergency_reports only (non-recursive).
    """
    global _observer
    with _observer_lock:
        if _observer is not None and _observer.is_alive():
            return  # already running

        observer = Observer()
        report_dir = Path(EMERGENCY_REPORT_DIR).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        observer.schedule(_UploadHandler(), str(report_dir), recursive=False)
        
        observer.daemon = True
        observer.start()
        _observer = observer


def watcher_is_active() -> bool:
    """Read observer status without creating another watcher."""
    with _observer_lock:
        return _observer is not None and _observer.is_alive()


def get_pending_changes() -> list[tuple[str, str]]:
    """
    Drain the change queue and return a list of (event_type, file_path) tuples.
    Non-blocking — returns immediately with whatever is in the queue.
    """
    changes = []
    while True:
        try:
            changes.append(_change_queue.get_nowait())
        except queue.Empty:
            break
    return changes


def stop_watcher() -> None:
    """Stop the observer (useful for clean shutdown in tests)."""
    global _observer
    with _observer_lock:
        if _observer and _observer.is_alive():
            _observer.stop()
            _observer.join()
            _observer = None
