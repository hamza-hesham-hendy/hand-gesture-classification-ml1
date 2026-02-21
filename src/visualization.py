"""
visualization.py
================
Hand-landmark visualization utilities.

Provides functions to plot individual hand samples (scatter + skeleton)
and to display multiple random samples from a DataFrame for quick
visual inspection of the dataset.
"""

import matplotlib.pyplot as plt
import numpy as np

# MediaPipe hand skeleton connections as pairs of 0-based landmark indices
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (5,6),(6,7),(7,8),              # Index
    (9,10),(10,11),(11,12),         # Middle
    (13,14),(14,15),(15,16),        # Ring
    (17,18),(18,19),(19,20),        # Pinky
    (0,5),(5,9),(9,13),(13,17),(0,17) # Wrist connections
]


def plot_hand_sample(row, title="Hand Sample"):
    """
    Plot a single hand sample from a DataFrame row.

    Draws the 21 landmarks as points and connects them with lines
    according to the standard hand-skeleton topology.

    Parameters
    ----------
    row : pd.Series
        A single row containing columns x1..x21 and y1..y21.
    title : str, optional
        Plot title (default "Hand Sample").
    """
    # Build arrays from columns x1..x21, y1..y21
    x = np.array([row[f"x{i}"] for i in range(1,22)])
    y = np.array([row[f"y{i}"] for i in range(1,22)])

    plt.figure(figsize=(4,4))
    plt.scatter(x, y)

    for connection in HAND_CONNECTIONS:
        x_coords = [x[connection[0]], x[connection[1]]]
        y_coords = [y[connection[0]], y[connection[1]]]
        plt.plot(x_coords, y_coords)

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.show()


def plot_multiple_samples(df, n_samples=5):
    """
    Plot multiple randomly chosen hand samples from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark columns (x1..x21, y1..y21) and a 'label' column.
    n_samples : int, optional
        Number of samples to plot (default 5).
    """
    samples = df.sample(n_samples)
    for idx, row in samples.iterrows():
        plot_hand_sample(row, title=f"Sample {idx} - Label: {row['label']}")
