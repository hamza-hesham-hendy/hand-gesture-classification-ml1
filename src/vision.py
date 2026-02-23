"""
vision.py
===================
Real-time hand-landmark detection and drawing utilities.

This module provides the interface with MediaPipe Tasks API for detecting 
hand landmarks in real-time. It includes logic for:
1. Setting up the MediaPipe HandLandmarker.
2. Extracting coordinates from video frames.
3. Rendering the hand skeleton on a frame using OpenCV.
4. Normalizing coordinates for consistency with the training data.
5. Implementing a 'GestureSmoother' to reduce flicker in predictions.
"""

import os
import cv2
import numpy as np
from collections import deque, Counter

import mediapipe as mp

# ──────────────────────────────────────────────
# New MediaPipe Tasks API (0.10.18+)
# ──────────────────────────────────────────────
BaseOptions         = mp.tasks.BaseOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

# ─── display constants ────────────────────────────────────────────────────────
_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 1.2
_FONT_COLOR  = (0, 255, 0)   # bright green label
_FONT_THICK  = 2
_BOX_COLOR   = (0, 0, 0)     # black background behind label

# Default path for the downloaded .task model
_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "hand_landmarker.task")

# Hand skeleton connections (pairs of landmark indices for drawing)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (5,6),(6,7),(7,8),              # Index
    (9,10),(10,11),(11,12),         # Middle
    (13,14),(14,15),(15,16),        # Ring
    (17,18),(18,19),(19,20),        # Pinky
    (0,5),(5,9),(9,13),(13,17),(0,17) # Wrist connections
]


# ──────────────────────────────────────────────
# Create a HandLandmarker (call once)
# ──────────────────────────────────────────────

def create_hand_landmarker(model_path=None, running_mode="VIDEO", num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
    """
    Create and return a MediaPipe HandLandmarker using the new Tasks API.

    This function initializes the MediaPipe object which uses a pre-trained
    deep learning model to find hand landmarks in image data.

    Parameters
    ----------
    model_path : str or None
        Path to `hand_landmarker.task`. Defaults to  models/hand_landmarker.task.
    running_mode : str
        "VIDEO" (default) or "LIVE_STREAM". 
    num_hands : int
        Maximum number of hands to detect (default 1).
    min_detection_confidence : float
        Sensitivity for initial hand detection (0.0 to 1.0).
    min_tracking_confidence : float
        Sensitivity for tracking a hand between frames (0.0 to 1.0).

    Returns
    -------
    HandLandmarker
        The configured MediaPipe detector object.
    """
    if model_path is None:
        model_path = _DEFAULT_MODEL

    mode_map = {
        "VIDEO": VisionRunningMode.VIDEO,
        "LIVE_STREAM": VisionRunningMode.LIVE_STREAM,
    }

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mode_map[running_mode],
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────
# Landmark extraction from a single frame
# ──────────────────────────────────────────────

def extract_landmarks_from_frame(frame, landmarker, timestamp_ms):
    """
    Extract hand landmarks from a single BGR frame.

    The output feature vector is in INTERLEAVED order:
        [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
    — exactly matching the training CSV column order.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (from cv2.VideoCapture).
    landmarker : HandLandmarker
        A landmarker created via create_hand_landmarker().
    timestamp_ms : int
        Frame timestamp in milliseconds (use frame_index * ms_per_frame).

    Returns
    -------
    landmarks_raw : np.ndarray (1, 63) or None
        Interleaved [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21], or None.
    result : HandLandmarkerResult or None
        Full result object (for drawing), or None.
    """
    # Convert BGR → RGB and wrap in mp.Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None, None

    # Use only the first detected hand
    hand_lms = result.hand_landmarks[0]   # list of 21 NormalizedLandmark

    # Build INTERLEAVED feature vector
    features = []
    for lm in hand_lms:
        features.extend([lm.x, lm.y, lm.z])

    landmarks_raw = np.array(features, dtype=np.float64).reshape(1, 63)
    return landmarks_raw, result


# ──────────────────────────────────────────────
# Draw landmarks on a frame (replaces mp_drawing)
# ──────────────────────────────────────────────

def draw_hand_landmarks(frame, result):
    """
    Draw the hand skeleton on *frame* using OpenCV.

    Parameters
    ----------
    frame : np.ndarray   (BGR)
    result : HandLandmarkerResult
        As returned by extract_landmarks_from_frame.
    """
    if result is None or not result.hand_landmarks:
        return

    h, w, _ = frame.shape

    for hand_lms in result.hand_landmarks:
        # Draw connections
        for (i, j) in _HAND_CONNECTIONS:
            pt1 = (int(hand_lms[i].x * w), int(hand_lms[i].y * h))
            pt2 = (int(hand_lms[j].x * w), int(hand_lms[j].y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmark dots
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

# ─── display helper ───────────────────────────────────────────────────────────

def put_label(frame, label):
    """Overlay *label* near the top-left corner with a dark background."""
    (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
    margin = 8
    x, y = 10, 50
    cv2.rectangle(
        frame,
        (x - margin, y - text_h - margin),
        (x + text_w + margin, y + baseline + margin),
        _BOX_COLOR,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame, label, (x, y),
        _FONT, _FONT_SCALE, _FONT_COLOR, _FONT_THICK, cv2.LINE_AA,
    )
    
# ──────────────────────────────────────────────
# Real-time normalization — reuses preprocessing.py
# ──────────────────────────────────────────────

# Column names matching the training CSV: x1,y1,z1, x2,y2,z2, ..., x21,y21,z21
_INTERLEAVED_COLS = [f"{ax}{i}" for i in range(1, 22) for ax in ("x", "y", "z")]


def normalize_landmarks_realtime(landmarks_array):
    """
    Normalize a single hand-landmark row for real-time inference.

    Delegates to preprocessing.normalize_landmarks() so that the exact same
    normalization logic is used at inference time as at training time.

    Parameters
    ----------
    landmarks_array : np.ndarray, shape (1, 63)
        Interleaved raw landmarks: [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21]

    Returns
    -------
    np.ndarray, shape (1, 63)
        Normalised, interleaved landmarks ready to be fed into a trained model.
    """
    import pandas as pd
    from preprocessing import normalize_landmarks

    df = pd.DataFrame(landmarks_array, columns=_INTERLEAVED_COLS)
    df_norm = normalize_landmarks(df)
    return df_norm.values


# ──────────────────────────────────────────────
# Smoothing helper
# ──────────────────────────────────────────────

class GestureSmoother:
    """
    Smooths per-frame gesture predictions by returning the MODE
    (most-frequent label) over a rolling window.

    Real-time predictions can often 'flicker' between labels due to noise.
    The smoother keeps a small history of the last N predictions and
    returns the one that occurred most often, creating a more stable output.
    """

    def __init__(self, window_size=15):
        """
        Initialize the smoother.
        
        Args:
            window_size (int): Number of previous frames to consider for voting.
        """
        self.window_size = window_size
        self._window = deque(maxlen=window_size)

    def update(self, prediction):
        """
        Add a new prediction to history and return the most common label.
        
        Args:
            prediction (str): The current frame's predicted gesture.
            
        Returns:
            str: The most frequent gesture in the current sliding window.
        """
        self._window.append(prediction)
        return Counter(self._window).most_common(1)[0][0]

    def reset(self):
        """Clear the history (useful when no hand is detected)."""
        self._window.clear()

