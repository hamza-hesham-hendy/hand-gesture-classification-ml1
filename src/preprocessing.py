"""
preprocessing.py
================
Data preprocessing utilities for hand-gesture classification.

This module handled the transformation of raw MediaPipe coordinates into a 
normalized format suitable for machine learning inference. It specifically
handles wrist-centering and scale-normalization to make the classifier 
invariant to hand position and size within the frame.

Key Features:
- Wrist-centering (moving the wrist landmark to the origin)
- Scale-normalization (scaling the hand by the palm length)
- Dataset splitting for training and testing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_landmarks(df):
    """
    Normalize hand landmarks for 1-based indexing (x1...x21):
    - Center x/y on wrist (landmark 1 -> index 0)
    - Scale x/y by middle finger tip (landmark 13 -> index 12)
    - Z columns are left completely untouched as requested.
    """
    df_norm = df.copy()

    # 1. Identify and Sort Columns Numerically
    # We strip the first character ('x') and convert the rest to int for sorting
    # This ensures 'x2' comes before 'x10'
    x_cols = sorted([c for c in df.columns if c.startswith('x')], key=lambda c: int(c[1:]))
    y_cols = sorted([c for c in df.columns if c.startswith('y')], key=lambda c: int(c[1:]))

    # Verify we have exactly 21 landmarks (x1...x21)
    if len(x_cols) != 21 or len(y_cols) != 21:
        raise ValueError(f"Expected 21 x and y columns, found {len(x_cols)} and {len(y_cols)}")

    # 2. Extract Data into Numpy Arrays
    # shape: (Rows, 21)
    X = df_norm[x_cols].values
    Y = df_norm[y_cols].values

    # 3. Stack into (Rows, 21, 2)
    # This securely pairs (x1, y1), (x2, y2), etc.
    coords = np.stack([X, Y], axis=-1)

    # 4. Center on Wrist
    # Wrist is now 'x1', which is at index 0 in our sorted list
    wrist = coords[:, 0, :][:, np.newaxis, :] 
    coords = coords - wrist

    # 5. Calculate Scale (Distance from Wrist to Middle Finger Tip)
    # Middle Finger Tip is 'x13', which is at index 12
    middle_tip = coords[:, 12, :] 
    scale = np.linalg.norm(middle_tip, axis=1)

    # Avoid division by zero
    scale[scale == 0] = 1e-6

    # 6. Apply Scale
    coords = coords / scale[:, np.newaxis, np.newaxis]

    # 7. Assign back to DataFrame
    df_norm[x_cols] = coords[:, :, 0]
    df_norm[y_cols] = coords[:, :, 1]

    return df_norm

def load_and_split(csv_path, test_size=0.2, random_state=42):
    """
    Load hand-landmark data from a CSV file, normalize it, and split
    into training and testing sets.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing hand landmarks and a 'label' column.
    test_size : float, optional
        Fraction of data to reserve for testing (default 0.2).
    random_state : int, optional
        Random seed for reproducible splits (default 42).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Feature matrices and label arrays for training and testing.
    """
    df = pd.read_csv(csv_path)
    
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column")

    df = normalize_landmarks(df)

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
    