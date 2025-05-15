import matplotlib.pyplot as plt
import numpy as np

def display_image(image: np.ndarray):
    """
    Display a NumPy array as an image in a Jupyter Notebook.

    Args:
        image (np.ndarray): The image to display.
    """
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()