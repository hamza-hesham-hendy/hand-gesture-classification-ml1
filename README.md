# ✋ Hand Gesture Recognition System

This project implements a full machine learning pipeline for recognizing **18 different hand gestures** using hand landmark coordinates. The system leverages **MediaPipe** for real-time landmark extraction and **Scikit-Learn** for high-accuracy classification.

The project covers data preprocessing, normalization, modular training with hyperparameter tuning (GridSearchCV), performance evaluation, and an **Interactive Demo** with both webcam and video processing support.

---

## 📁 Project Structure

```text
hand-gesture-recognition/
├── data/
│   └── hand_landmarks_data.csv     # Pre-extracted landmarks (18 classes)
├── models/
│   ├── RandomForest.pkl            # Serialized ML model
│   ├── SVM.pkl                     # Serialized ML model
│   ├── LogisticRegression.pkl      # Serialized ML model
│   └── hand_landmarker.task        # MediaPipe landmarker weights
├── src/
│   ├── preprocessing.py            # Coordinate normalization logic
│   ├── visualization.py            # Landmark & skeleton plotting
│   ├── train.py                    # GridSearchCV training logic
│   ├── evaluation.py               # Metrics & Confusion Matrices
│   ├── vision.py                   # MediaPipe abstraction layer
│   └── inference.py                # Core demo and prediction logic
├── HandGestureSystem.ipynb         # Main Jupyter Notebook (Training & Demo)
├── requirements.txt                # Project dependencies
├── setup_env.bat                   # Environment setup script
└── README.md                       # Project documentation
```

---

## 🧪 Captured Gestures (18 Classes)
The system is trained to recognize a wide variety of gestures, including:
*   **Signs:** `ok`, `stop`, `palm`, `fist`, `like`, `dislike`, `call`, `rock`, `mute`.
*   **Numbers/Fingers:** `one`, `two_up`, `three`, `four`, `peace`.
*   **Inverted Gestures:** `peace_inverted`, `stop_inverted`, `two_up_inverted`.

---

## 🛠️ Components & Features

### 1. Advanced Normalization (`src/preprocessing.py`)
To ensure robustness against hand size and position, the system:
- **Centering:** Re-centers all landmarks relative to the wrist (landmark 0).
- **Scale Invariance:** Normalizes the distance of all landmarks based on the length from the wrist to the middle-finger base.

### 2. Machine Learning Pipeline (`src/train.py`)
We train and compare three distinct models using **GridSearchCV** for best parameter selection:
- **Random Forest:** Ensemble-based classifier.
- **SVM (RBF):** Highly reliable for non-linear boundary detection.
- **Logistic Regression:** Provides a strong linear baseline.

### 3. Smart Interactive Demo (`src/inference.py`)
The demo offers real-time visualization with several features:
- **Source Choice:** Run using a **Live Webcam** or process a local **Video File** (`my_hand_video.mp4`).
- **Best Model Detection:** Automatically detects which saved model has the highest validation score and uses it for prediction.
- **Voting System:** Combines the predictions of all three models (RandomForest, SVM, LogReg) for a consensus-based stable result.
- **Gesture Smoothing:** Implements a rolling-window filter to eliminate "flicker" in real-time predictions.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training & Evaluation
Open `HandGestureSystem.ipynb` in your favorite editor (VS Code, Jupyter, etc.).
- Run the **Preprocessing** and **Visualization** sections to explore the data.
- Run the **Training** section to generate the `.pkl` files in the `models/` folder.
- Run the **Evaluation** section to see detailed metrics and confusion matrices.

### 3. Running the Demo
Scroll to the final cell in the notebook. The demo will interactively ask you for:
1.  **Input Source:** Press `L` for Webcam or `V` for Video.
2.  **Prediction Method:** 
    *   Press `B` for the **Best Model** (Auto-detected).
    *   Press `M` for **Majority Voting** (Ensemble of all models).
3.  **Save Output:** Option to save the result as `annotated_output.mp4`.

---

## 📊 Performance Summary

| Model | Accuracy | F1-Score | Precision | Recall | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 98.0% | 0.98 | 0.98 | 0.98 | 0.9991 |
| SVM | 99.0% | 0.99 | 0.99 | 0.99 | 0.9997 |
| Logistic Regression | 91.7% | 0.92 | 0.92 | 0.92 | 0.996 |

---

## 🏆 Model Choice Rationale
After comparing all experiments, the **SVM** model was selected for the final demo due to:
1. **Superior Accuracy**: Achieved highest accuracy on the test set.
2. **Robustness**: High F1-scores across all 18 classes, indicating it handles both frequent and rare gestures well.
3. **Generalization**: The gap between cross-validation scores and test scores was minimal, showing low variance.
4. **Latency**: Despite being a complex model, the prediction time remains well within the requirements for real-time (30+ FPS) processing.

---

## 👨‍💻 Author
**Hamza Hesham Hendy**  
*Hand Gesture Recognition - Machine Learning Project*
