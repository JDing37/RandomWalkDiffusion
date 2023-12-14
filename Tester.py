import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to generate random data for each frame
def generate_data(frame):
    np.random.seed(frame)
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    return x, y


# Create a figure and axis for the animation
fig, ax = plt.subplots()

# Create an initial empty 2D histogram
hist = ax.hist2d([], [], bins=(30, 30), cmap='Blues')


# Function to update the plot for each frame
def update(frame, n):
    x, y = generate_data(frame)
    print(n)
    # Clear the previous histogram
    ax.clear()

    # Create a new histogram
    hist = ax.hist2d(x, y, bins=(30, 30), cmap='Blues')


# Create the animation
animation = FuncAnimation(fig, update, frames=80, interval=150, repeat=True)

# Show the plot
plt.show()
