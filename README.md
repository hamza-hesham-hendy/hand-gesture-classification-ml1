# ✋ Hand Gesture Classification using MediaPipe Landmarks

This project builds a machine learning pipeline to classify hand gestures using hand landmark coordinates extracted from the **HaGRID** (Hand Gesture Recognition Image) dataset.

The system includes data preprocessing, visualization, training multiple ML models, evaluation with performance metrics, and a real-time video demo using MediaPipe.

---

## 📁 Project Structure

```
hand-gesture-classification-ml1/
│
├── data/
│   └── hand_landmarks_data.csv     # HaGRID landmarks (18 classes, 21 landmarks × 3 coords)
│
├── models/
│   ├── RandomForest.pkl            # Trained Random Forest model
│   ├── SVM.pkl                     # Trained SVM model
│   ├── LogisticRegression.pkl      # Trained Logistic Regression model
│   └── hand_landmarker.task        # MediaPipe hand landmarker model
│
├── src/
│   ├── preprocessing.py            # Normalization and train/test split
│   ├── visualization.py            # Hand landmark plotting
│   ├── train.py                    # Model training with GridSearchCV
│   ├── evaluation.py               # Metrics and confusion matrices
│   └── image_processing.py         # Real-time MediaPipe landmark detection
│
├── testing.ipynb                   # End-to-end notebook (load → train → evaluate)
├── video_demo.py                   # Real-time gesture classification on video
├── annotated_output.mp4            # Sample output video
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ✅ Components

### 1️⃣ Data Preprocessing (`src/preprocessing.py`)
- **Wrist-centering**: x/y coordinates are re-centered so the wrist (landmark 1) is the origin.
- **Scale normalization**: x/y are divided by the distance from the wrist to the middle-finger tip (landmark 13), making all hands scale-invariant.
- **Z coordinates** are left untouched (already processed by MediaPipe).
- Stratified **train/test split** (80/20).

### 2️⃣ Visualization (`src/visualization.py`)
- Plot individual hand samples as scatter + skeleton overlays.
- Plot multiple random samples from the dataset for visual inspection.

### 3️⃣ Model Training (`src/train.py`)
Three ML classifiers trained with **GridSearchCV** hyperparameter tuning:

| Model | Best Accuracy |
|---|---|
| 🌲 Random Forest | **97.76%** |
| 🔵 SVM (RBF Kernel) | 93.34% |
| 📈 Logistic Regression | 85.76% |

### 4️⃣ Evaluation (`src/evaluation.py`)
For each model, the following metrics are reported:
- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **F1-score** (weighted)
- **AUC** (one-vs-rest)
- **Confusion Matrix** visualization

### 5️⃣ Notebook (`testing.ipynb`)
End-to-end pipeline in a single executable notebook:
1. Load and visualize the dataset
2. Preprocess (normalize landmarks)
3. Train all 3 models
4. Evaluate and compare models
5. Save trained models to `models/`

### 6️⃣ Video Demo (`video_demo.py`)
Real-time hand-gesture classification on video or webcam:
- Uses **MediaPipe Tasks API** (`HandLandmarker`) for landmark extraction.
- Supports **majority voting** across all 3 models or a single specified model.
- **Mode-smoothing** over a rolling time window to stabilize predictions.
- Draws hand skeleton overlay and gesture label on each frame.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/hamza-hesham-hendy/hand-gesture-classification-ml1.git
cd hand-gesture-classification-ml1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Run the Notebook (Training + Evaluation)

```bash
jupyter notebook testing.ipynb
```

Run all cells to:
- Load and preprocess the dataset
- Train Random Forest, SVM, and Logistic Regression models
- Evaluate each model with metrics and confusion matrices
- Save trained models to `models/`

### Option 2: Run the Video Demo

**With an input video file:**
```bash
python video_demo.py --input path/to/video.mp4 --output annotated_output.mp4
```

**With live webcam (default camera):**
```bash
python video_demo.py
```

**Using a specific model instead of majority voting:**
```bash
python video_demo.py --input video.mp4 --use-model RandomForest
```

**All available arguments:**
| Argument | Description | Default |
|---|---|---|
| `--input` | Path to input video file (omit for webcam) | Webcam (index 0) |
| `--output` | Save annotated output video to this path | None |
| `--models-dir` | Folder containing trained `.pkl` files | `models/` |
| `--window` | Mode-smoothing window size (frames) | 15 |
| `--use-model` | Use a single model (e.g., `RandomForest`, `SVM`) | Majority vote |

Press **`q`** to quit the live preview window.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **Random Forest** | **97.76%** | **97.80%** | **97.76%** | **97.76%** | **99.98%** |
| SVM (RBF) | 93.34% | 93.60% | 93.34% | 93.34% | 99.72% |
| Logistic Regression | 85.76% | 85.82% | 85.76% | 85.70% | 99.10% |

**Best Model: Random Forest** with 97.76% accuracy.

---

## 🧠 Models Used

- **Random Forest** — Ensemble of decision trees with hyperparameter search over `n_estimators`, `max_depth`, and `criterion`.
- **SVM (RBF Kernel)** — Support Vector Machine with search over `C` and `gamma`.
- **Logistic Regression** — Linear classifier with L2 regularization and search over `C`.

All models are tuned via **3-fold cross-validation** using `GridSearchCV`.

---

## 👨‍💻 Author

**Hamza Hesham Hendy**

Hand Gesture Classification – ML1 Project
