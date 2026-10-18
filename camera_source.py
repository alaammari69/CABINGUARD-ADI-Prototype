import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraSource:
    """Owns the webcam. Grabs frames in a background thread and keeps only the newest one,
    so several detectors can read the same camera without fighting over it."""

    def __init__(self, cam_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.cam_index: int = cam_index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps

        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()  # protects the three fields below

        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0         # increases by 1 for every new frame
        self._frame_time: float = 0.0   # capture time in seconds since the camera opened

    def start(self) -> None:
        """Start the capture thread. Returns immediately."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to exit and wait until it has finished."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def is_running(self) -> bool:
        """False once the capture thread has ended (camera failed or stop() was called)."""
        return self._thread is not None and self._thread.is_alive()

    def get(self, last_id: int = -1) -> Tuple[Optional[np.ndarray], int, float]:
        """Newest frame, but only if it is newer than `last_id`.

        returns: (frame, frame_id, capture_time_s)
                 frame is None if there is no new frame yet. Otherwise it is a private COPY,
                 so each detector can draw on it without affecting the others.
        """
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, self._frame_id, self._frame_time
            return self._frame.copy(), self._frame_id, self._frame_time

    def _loop(self) -> None:
        """Capture loop. Runs in the background thread (internal use)."""
        cap: cv2.VideoCapture = cv2.VideoCapture(self.cam_index)  # if no image appears, try 1
        if not cap.isOpened():
            print("Could not open camera")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # set first: allows HD at 30 fps over USB
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # The camera silently picks the nearest mode it supports, so print what we really got
        print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
              f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

        start: float = time.time()
        try:
            while not self._stop_event.is_set():
                ok: bool
                frame: np.ndarray
                ok, frame = cap.read()
                if not ok:
                    print("Could not read from camera")
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._frame_time = time.time() - start
        finally:
            cap.release()  # always free the camera