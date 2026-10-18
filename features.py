import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, List, Optional, Tuple

import numpy as np

Landmarks = List[Any]  # MediaPipe NormalizedLandmark list (478 items)

# --- FaceMesh landmark indices ---
RIGHT_EYE: List[int] = [33, 160, 158, 133, 153, 144]
LEFT_EYE: List[int] = [362, 385, 387, 263, 373, 380]
NOSE_TIP: int = 1
EYE_OUTER_R: int = 33      # outer corner of the person's right eye
EYE_OUTER_L: int = 263     # outer corner of the person's left eye
MOUTH_CORNER_R: int = 61
MOUTH_CORNER_L: int = 291


# ======================= per-frame signals =======================

@dataclass
class FrameSignals:
    """Raw measurements from ONE frame. None means 'not available'."""
    t: float                       # time in seconds
    face_found: bool
    ear: Optional[float]           # eye aspect ratio
    pitch: Optional[float]         # head nod angle, degrees
    yaw: Optional[float]           # head turn angle, degrees
    roll: Optional[float]          # head tilt angle, degrees
    asymmetry: Optional[float]     # signed mouth-corner height difference (see below)
    nose_x: Optional[float]        # nose position in inter-eye-distance units
    nose_y: Optional[float]


def _px(points: Landmarks, i: int, w: int, h: int) -> np.ndarray:
    """Landmark i as a pixel position [x, y]."""
    return np.array([points[i].x * w, points[i].y * h])


def eye_aspect_ratio(points: Landmarks, idx: List[int], w: int, h: int) -> float:
    """EAR for one eye (6 landmark indices)."""
    p: List[np.ndarray] = [_px(points, i, w, h) for i in idx]
    vertical: float = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    horizontal: float = float(np.linalg.norm(p[0] - p[3]))
    return vertical / (2.0 * horizontal)


def compute_asymmetry(points: Landmarks, w: int, h: int) -> float:
    """Difference in height of the two mouth corners, measured perpendicular to
    the eye line (so head tilt doesn't fake it) and divided by eye distance
    (so camera distance doesn't matter). ~0 for a symmetric face.
    Positive/negative just tells which side is lower."""
    r_eye: np.ndarray = _px(points, EYE_OUTER_R, w, h)
    l_eye: np.ndarray = _px(points, EYE_OUTER_L, w, h)
    eye_vec: np.ndarray = l_eye - r_eye
    iod: float = float(np.linalg.norm(eye_vec))      # inter-eye distance
    u: np.ndarray = eye_vec / iod                    # unit vector along the eye line
    n: np.ndarray = np.array([-u[1], u[0]])          # unit vector perpendicular to it
    mid: np.ndarray = (r_eye + l_eye) / 2.0
    # signed distance of each mouth corner from the eye line, along n
    h_r: float = float(np.dot(_px(points, MOUTH_CORNER_R, w, h) - mid, n))
    h_l: float = float(np.dot(_px(points, MOUTH_CORNER_L, w, h) - mid, n))
    return (h_r - h_l) / iod


def head_angles_from_matrix(m: np.ndarray) -> Tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from MediaPipe's 4x4 face transform matrix.
    NOTE: axis conventions/signs are from my understanding, verify by testing:
    nod and watch which value moves."""
    r: np.ndarray = m[:3, :3]
    r = r / np.linalg.norm(r, axis=0)  # remove any scale baked into the matrix
    pitch: float = math.degrees(math.atan2(r[2, 1], r[2, 2]))       # rotation about x (nod)
    yaw: float = math.degrees(-math.asin(float(np.clip(r[2, 0], -1.0, 1.0))))  # about y (turn)
    roll: float = math.degrees(math.atan2(r[1, 0], r[0, 0]))        # about z (tilt)
    return pitch, yaw, roll


def build_frame_signals(t: float, result: Any, w: int, h: int) -> FrameSignals:
    """Turn one MediaPipe result into a FrameSignals."""
    if not result.face_landmarks:
        return FrameSignals(t, False, None, None, None, None, None, None, None)

    points: Landmarks = result.face_landmarks[0]
    ear: float = (eye_aspect_ratio(points, RIGHT_EYE, w, h)
                  + eye_aspect_ratio(points, LEFT_EYE, w, h)) / 2.0

    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None
    if result.facial_transformation_matrixes:  # only present if enabled in the options
        pitch, yaw, roll = head_angles_from_matrix(np.array(result.facial_transformation_matrixes[0]))

    iod: float = float(np.linalg.norm(_px(points, EYE_OUTER_L, w, h) - _px(points, EYE_OUTER_R, w, h)))
    nose: np.ndarray = _px(points, NOSE_TIP, w, h)
    return FrameSignals(t=t, face_found=True, ear=ear, pitch=pitch, yaw=yaw, roll=roll,
                        asymmetry=compute_asymmetry(points, w, h),
                        nose_x=float(nose[0] / iod), nose_y=float(nose[1] / iod))


# ======================= window features =======================

@dataclass
class WindowFeatures:
    """Features for one sliding window. This is what the classifier sees."""
    mean_ear: float
    min_ear: float
    closed_fraction: float            # fraction of face frames with EAR below threshold
    face_lost_fraction: float         # fraction of ALL frames with no face
    longest_lost_s: float             # longest run without a face (approximate)
    time_since_face_s: float          # how long since a face was last seen
    pitch_from_baseline: float        # mean pitch minus normal-posture baseline
    pitch_range: float                # max - min pitch in the window
    max_pitch_speed: float            # fastest nod, degrees per second
    motion_speed: float               # mean nose speed, inter-eye-distances per second
    band_power_ratio: float           # share of motion power in the 3-8 Hz band (seizure cue)
    dominant_freq_hz: float           # strongest motion frequency
    mean_abs_asymmetry: float
    asymmetry_std: float

    def to_vector(self) -> List[float]:
        """Fixed-order list of numbers for scikit-learn."""
        return [float(v) for v in asdict(self).values()]


class WindowFeatureExtractor:
    """Keeps the last `window_s` seconds of FrameSignals and computes features."""

    def __init__(self, window_s: float = 2.0, ear_closed_thresh: float = 0.2,
                 band_hz: Tuple[float, float] = (3.0, 8.0), min_frames: int = 10) -> None:
        self.window_s: float = window_s
        self.ear_closed_thresh: float = ear_closed_thresh  # per-person, needs calibration
        self.band_hz: Tuple[float, float] = band_hz
        self.min_frames: int = min_frames
        self.baseline_pitch: float = 0.0                   # normal driving posture
        self._buf: Deque[FrameSignals] = deque()

    def set_baseline(self, pitch: float) -> None:
        """Call with the driver's pitch while they sit normally."""
        self.baseline_pitch = pitch

    def add(self, s: FrameSignals) -> None:
        """Add a frame and drop frames older than the window."""
        self._buf.append(s)
        while self._buf and s.t - self._buf[0].t > self.window_s:
            self._buf.popleft()

    def compute(self) -> Optional[WindowFeatures]:
        """Features for the current window, or None if too few frames yet."""
        frames: List[FrameSignals] = list(self._buf)
        n: int = len(frames)
        if n < self.min_frames:
            return None

        found: List[FrameSignals] = [f for f in frames if f.face_found]
        face_lost_fraction: float = 1.0 - len(found) / n

        # --- face lost duration ---
        longest_lost: float = 0.0
        run_start: Optional[float] = None
        for f in frames:
            if not f.face_found:
                if run_start is None:
                    run_start = f.t
                longest_lost = max(longest_lost, f.t - run_start)
            else:
                run_start = None
        time_since_face: float = frames[-1].t - found[-1].t if found else self.window_s

        if len(found) < self.min_frames:
            # not enough face data: report loss, zeros elsewhere
            return WindowFeatures(0.0, 0.0, 0.0, face_lost_fraction, longest_lost, time_since_face,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        t: np.ndarray = np.array([f.t for f in found])

        # --- eyes ---
        ears: np.ndarray = np.array([f.ear for f in found], dtype=float)
        closed_fraction: float = float(np.mean(ears < self.ear_closed_thresh))

        # --- head pitch (skipped if the matrix was not enabled) ---
        pitch_from_baseline: float = 0.0
        pitch_range: float = 0.0
        max_pitch_speed: float = 0.0
        if all(f.pitch is not None for f in found):
            pitch: np.ndarray = np.degrees(np.unwrap(np.radians([f.pitch for f in found])))
            pitch_from_baseline = float(pitch.mean() - self.baseline_pitch)
            pitch_range = float(pitch.max() - pitch.min())
            dt: np.ndarray = np.maximum(np.diff(t), 1e-3)
            max_pitch_speed = float(np.max(np.abs(np.diff(pitch) / dt)))

        # --- head motion: speed and frequency of the nose position ---
        x: np.ndarray = np.array([f.nose_x for f in found], dtype=float)
        y: np.ndarray = np.array([f.nose_y for f in found], dtype=float)
        dt = np.maximum(np.diff(t), 1e-3)
        motion_speed: float = float(np.mean(np.hypot(np.diff(x), np.diff(y)) / dt))
        band_ratio, dom_freq = self._band_power([x, y], t)

        # --- asymmetry ---
        asym: np.ndarray = np.array([f.asymmetry for f in found], dtype=float)

        return WindowFeatures(
            mean_ear=float(ears.mean()), min_ear=float(ears.min()),
            closed_fraction=closed_fraction, face_lost_fraction=face_lost_fraction,
            longest_lost_s=longest_lost, time_since_face_s=time_since_face,
            pitch_from_baseline=pitch_from_baseline, pitch_range=pitch_range,
            max_pitch_speed=max_pitch_speed, motion_speed=motion_speed,
            band_power_ratio=band_ratio, dominant_freq_hz=dom_freq,
            mean_abs_asymmetry=float(np.mean(np.abs(asym))), asymmetry_std=float(asym.std()))

    def _band_power(self, signals: List[np.ndarray], t: np.ndarray) -> Tuple[float, float]:
        """(share of power in band_hz, dominant frequency in Hz) of the signals."""
        duration: float = float(t[-1] - t[0])
        if len(t) < 16 or duration <= 0:
            return 0.0, 0.0
        fs: float = (len(t) - 1) / duration                 # actual sampling rate
        grid: np.ndarray = np.arange(t[0], t[-1], 1.0 / fs)  # evenly spaced times
        total: Optional[np.ndarray] = None
        for sig in signals:
            u: np.ndarray = np.interp(grid, t, sig)          # resample to even spacing
            u = (u - u.mean()) * np.hanning(len(u))          # remove offset, smooth edges
            p: np.ndarray = np.abs(np.fft.rfft(u)) ** 2
            total = p if total is None else total + p
        freqs: np.ndarray = np.fft.rfftfreq(len(grid), d=1.0 / fs)
        above_drift: np.ndarray = freqs > 0.5                # ignore slow drifting
        hi: float = min(self.band_hz[1], 0.5 * fs)           # can't see above Nyquist
        in_band: np.ndarray = (freqs >= self.band_hz[0]) & (freqs <= hi)
        denom: float = float(total[above_drift].sum())
        if denom < 1e-12:
            return 0.0, 0.0
        return float(total[in_band].sum() / denom), float(freqs[above_drift][np.argmax(total[above_drift])])