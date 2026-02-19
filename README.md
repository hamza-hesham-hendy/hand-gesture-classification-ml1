
---

# 📄 README.md

# ✋ Hand Gesture Classification using MediaPipe Landmarks
```markdown

This project builds a machine learning pipeline to classify hand gestures using hand landmark coordinates extracted from the HaGRID dataset.

The system includes preprocessing, visualization, training multiple ML models, and evaluation with performance metrics.
```

---
## 📁 Project Structure
````
hand-gesture-classification-ml1/
│
├── .gitignore
│
├── data/
│   └── hand_landmarks_data.csv
│
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── train.py
│   ├── evaluation.py
│   ├── image_processing.py
│   └── utils.py
│
├── testing.ipynb
├── video_demo.py
├── README.md
└── requirements.txt

````

---

## ✅ Completed Components

### 1️⃣ Data Preprocessing (`preprocessing.py`)
- Landmark normalization (centered on wrist)
- Scaling based on hand size
- Train/Test split

---

### 2️⃣ Visualization (`visualization.py`)
- Plot a single hand sample
- Plot multiple gesture samples
- Useful for sanity checking dataset quality

---

### 3️⃣ Model Training (`train.py`)
Implemented 3 machine learning models:

- 🌲 Random Forest
- 🔵 Support Vector Machine (RBF Kernel)
- 📈 Logistic Regression

All models are trained with good default parameters (debug mode for fast training).

---

### 4️⃣ Evaluation (`evaluation.py`)
For each model:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion Matrix visualization

---

### 5️⃣ Testing Notebook (`testing.ipynb`)
End-to-end pipeline:
1. Load dataset
2. Preprocess
3. Train models
4. Evaluate models

---

## ⏳ Future Work

- 🎥 Real-time gesture prediction from video (`video_demo.py`)
- 🖐 Landmark extraction using MediaPipe (`image_processing.py`)
- 📊 Experiment tracking using MLflow (research branch)
- 📄 Improve documentation and add results summary

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/hamza-hesham-hendy/hand-gesture-classification-ml1.git
cd hand-gesture-classification-ml1
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Open the notebook:

```bash
jupyter notebook testing.ipynb
```

Run all cells to train and evaluate models.

---

## 🧠 Models Used

* Random Forest
* Support Vector Machine (RBF)
* Logistic Regression

---

## 📌 Branch Strategy

* `main` → Clean ML pipeline (no experiment tracking)
* `research` → MLflow experiments and model tracking

---

## 👨‍💻 Author

Hand Gesture Classification – ML1 Project

````
Hamza Hesham Hendy
````
---
