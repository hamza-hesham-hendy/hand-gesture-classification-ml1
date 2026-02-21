"""
video_demo.py
=============
Run hand-gesture classification on a video file or live webcam stream.
All trained models vote per-frame (majority voting), and the winning label
is further stabilised with a mode-smoothing window over time.

Usage — input video file:
    python video_demo.py --input path/to/video.mp4

Usage — live webcam (default camera):
    python video_demo.py

Usage — save annotated output:
    python video_demo.py --input video.mp4 --output out.mp4

Models are loaded automatically from the  models/  folder.
Press 'q' to quit the live preview window.
"""

import argparse
import os
import pickle
import sys

import cv2
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─── add src/ to the Python path ──────────────────────────────────────────────
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

from image_processing import (
    GestureSmoother,
    create_hand_landmarker,
    draw_hand_landmarks,
    extract_landmarks_from_frame,
    normalize_landmarks_realtime,
    put_label,
)

# ─── model helpers ────────────────────────────────────────────────────────────

def load_models(models_dir="models"):
    """
    Load every  .pkl  file found in *models_dir* and return them as a dict.
    """
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"Models directory not found: {models_dir!r}\n"
            "Please run the notebook and save your models first:\n"
            "    from train import save_models\n"
            "    save_models(models, folder='models/')"
        )

    models = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            name = os.path.splitext(fname)[0]
            path = os.path.join(models_dir, fname)
            with open(path, "rb") as fh:
                models[name] = pickle.load(fh)
            print(f"  Loaded: {name}  ({path})")

    if not models:
        raise FileNotFoundError(
            f"No .pkl model files found in {models_dir!r}."
        )
    return models


def majority_vote(models, features):
    """
    Ask every model to predict and return the majority-vote label.
    """
    votes = [model.predict(features)[0] for model in models.values()]
    return Counter(votes).most_common(1)[0][0]


# ─── main processing loop ─────────────────────────────────────────────────────

def process_video(models, input_source, output_path=None, window_size=15, use_model_name=None):
    """
    Classify hand gestures on every frame using majority voting or a single specified model,
    and smoothing.
    """
    if use_model_name:
        if use_model_name not in models:
            raise ValueError(f"Specified model '{use_model_name}' not found among loaded models.")
        print(f"  -> Using only model: '{use_model_name}'")
    else:
        print(f"  -> {len(models)} model(s) will vote per frame: {list(models.keys())}")

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source!r}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    smoother = GestureSmoother(window_size=window_size)

    # Create the MediaPipe HandLandmarker (new Tasks API)
    landmarker = create_hand_landmarker(running_mode="VIDEO")

    frame_idx = 0
    ms_per_frame = 1000.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * ms_per_frame)

        # ── Step 1: extract raw landmarks ─────────────────────────────────
        landmarks_raw, result = extract_landmarks_from_frame(
            frame, landmarker, timestamp_ms
        )

        if landmarks_raw is None:
            display_label = "No hand detected"
            smoother.reset()

        else:
            # ── Step 2: normalise (reuses preprocessing.normalize_landmarks) ──
            landmarks_norm = normalize_landmarks_realtime(landmarks_raw)

            # ── Step 3: Predict with single model or majority vote ────────
            if use_model_name:
                # Use the specified single model
                predicted_label = models[use_model_name].predict(landmarks_norm)[0]
            else:
                # Default to majority voting
                predicted_label = majority_vote(models, landmarks_norm)

            # ── Step 4: smooth — mode over a rolling time window ──────────
            display_label = smoother.update(predicted_label)

            # ── Step 5: draw hand skeleton on the frame ───────────────────
            draw_hand_landmarks(frame, result)

        # ── Step 6: overlay gesture label ─────────────────────────────────
        put_label(frame, display_label)

        # ── Step 7: write annotated frame to output file ───
        if writer:
            writer.write(frame)

        # ── Step 8: live preview — press 'q' to quit ──────────────────────
        cv2.imshow("Hand Gesture Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    landmarker.close()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("Done.")
    if output_path:
        print(f"Annotated video saved -> {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Hand-gesture demo: majority vote across all models + time smoothing.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Folder containing trained .pkl model files (default: models/).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to an input video file.  Omit to use the default webcam (index 0).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(Optional) Save annotated output video here, e.g.  output.mp4",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Mode-smoothing window size in frames (default: 15).",
    )
    parser.add_argument(
        "--use-model",
        type=str,
        default=None,
        help="(Optional) Specify a single model name (e.g., 'RandomForest', 'SVM') to use instead of majority voting."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"Loading models from: {args.models_dir}/")
    models = load_models(args.models_dir)

    input_source = args.input if args.input else 0

    process_video(
        models=models,
        input_source=input_source,
        output_path=args.output,
        window_size=args.window,
        use_model_name=args.use_model,
    )