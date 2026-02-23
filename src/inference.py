"""
inference.py
============
Core logic for running hand-gesture classification.

This script acts as the main entry point for running the gesture recognition
system on either pre-recorded video or a live webcam feed. It ties together
landmark detection, normalization, and model prediction.
"""

import os
import pickle
import sys
import cv2
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─── add current dir (src/) to the Python path ─────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision import (
    GestureSmoother,
    create_hand_landmarker,
    draw_hand_landmarks,
    extract_landmarks_from_frame,
    normalize_landmarks_realtime,
    put_label,
)

# ─── model helpers ────────────────────────────────────────────────────────────

def load_models(models_dir=None):
    """
    Load every .pkl file found in *models_dir* and return them as a dict.
    """
    if models_dir is None:
        # If we are in src/, models is one level up
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"Models directory not found: {models_dir!r}\n"
            "Please run the notebook and save your models first."
        )

    models = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            name = os.path.splitext(fname)[0]
            path = os.path.join(models_dir, fname)
            with open(path, "rb") as fh:
                models[name] = pickle.load(fh)
            print(f"  Loaded model: {name}")

    if not models:
        raise FileNotFoundError(f"No .pkl model files found in {models_dir!r}.")
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
    Classify hand gestures on every frame using majority voting or a single specified model.
    """
    if use_model_name:
        if use_model_name not in models:
            raise ValueError(f"Specified model '{use_model_name}' not found.")
        print(f"  -> Using only model: '{use_model_name}'")
    else:
        print(f"  -> Running with {len(models)} models: {list(models.keys())}")

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
    landmarker = create_hand_landmarker(running_mode="VIDEO")

    frame_idx = 0
    ms_per_frame = 1000.0 / fps

    print("Processing... Press 'q' in the window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        timestamp_ms = int(frame_idx * ms_per_frame)
        landmarks_raw, result = extract_landmarks_from_frame(frame, landmarker, timestamp_ms)

        if landmarks_raw is None:
            display_label = "No hand detected"
            smoother.reset()
        else:
            landmarks_norm = normalize_landmarks_realtime(landmarks_raw)
            if use_model_name:
                predicted_label = models[use_model_name].predict(landmarks_norm)[0]
            else:
                predicted_label = majority_vote(models, landmarks_norm)
            
            display_label = smoother.update(predicted_label)
            draw_hand_landmarks(frame, result)

        put_label(frame, display_label)
        if writer: writer.write(frame)

        cv2.imshow("Hand Gesture Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    landmarker.close()
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("Simulation ended.")


def run_gesture_demo(models_dir=None, window_size=15):
    """
    Interactive function to run the gesture demo. 
    Asks the user for input source (live/video) and saving preference.
    """
    print("\n--- Hand Gesture Demo Setup ---")
    choice = input("Run (L)ive webcam or (V)ideo file? [L/V]: ").strip().upper()
    
    input_source = 0
    if choice == 'V':
        # Assume file exists in root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_source = os.path.join(root_dir, "my_hand_video.mp4")
        if not os.path.exists(input_source):
            print(f"Error: {input_source} not found. Defaulting to webcam.")
            input_source = 0
        else:
            print(f"Using video file: {input_source}")
    else:
        print("Using live webcam.")

    save_choice = input("Save annotated output? (Y/N): ").strip().upper()
    output_path = None
    if save_choice == 'Y':
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(root_dir, "annotated_output.mp4")
        print(f"Output will be saved to: {output_path}")

    # Load models
    models = load_models(models_dir)
    
    # helper to find best model name based on best_score_ (validation accuracy)
    best_model_name = None
    max_score = -1
    for name, model in models.items():
        # Check if it's a GridSearchCV object
        score = getattr(model, 'best_score_', -1)
        if score > max_score:
            max_score = score
            best_model_name = name
            
    if not best_model_name:
        # Fallback to RandomForest if we can't determine
        best_model_name = 'RandomForest' if 'RandomForest' in models else list(models.keys())[0]

    print("\nPrediction Method:")
    method_choice = input(f"Use (B)est model ({best_model_name}) or (M)ajority voting? [B/M]: ").strip().upper()
    
    use_model_name = None
    if method_choice == 'B':
        use_model_name = best_model_name
        print(f"  -> Selected: {best_model_name} (Best)")
    else:
        print("  -> Selected: Majority Voting")

    # Run processing
    process_video(
        models=models,
        input_source=input_source,
        output_path=output_path,
        window_size=window_size,
        use_model_name=use_model_name
    )