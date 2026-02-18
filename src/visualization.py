import matplotlib.pyplot as plt
import numpy as np

# Adjusted for x1..x21 columns (convert to 0-based indexing)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (5,6),(6,7),(7,8),              # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (13,14),(14,15),(15,16),        # Ring
    (17,18),(18,19),(19,20),        # Pinky
    (0,5),(5,9),(9,13),(13,17),(0,17) # Wrist connections
]

def plot_hand_sample(row, title="Hand Sample"):
    """
    Plot a single hand sample from a dataframe row.
    Assumes x1..x21, y1..y21 exist.
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
    Plot multiple random hand samples.
    """
    samples = df.sample(n_samples)
    for idx, row in samples.iterrows():
        plot_hand_sample(row, title=f"Sample {idx} - Label: {row['label']}")
