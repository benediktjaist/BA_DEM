import pygame
import os
import numpy as np
import imageio

# Initialize Pygame
pygame.init()

# Set up the window dimensions
win_width = 816
win_height = 624

# Set up the frame rate and clock
FPS = 30
clock = pygame.time.Clock()

# Set up the colors
BLACK = (0, 0, 0)

# Set up the positions of the circle center
positions = [(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)]

# Set up the time steps
time_steps = [0, 0.5, 1, 1.5, 2]

# Calculate the number of frames needed for the desired duration
duration = time_steps[-1] - time_steps[0]
num_frames = int(duration * FPS)

# Set up the video writer
video_dir = "C:/Users/Jaist/Desktop/ba_videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
video_file = os.path.join(video_dir, "animation.mp4")
fps = FPS

# Set up the Pygame window
win = pygame.display.set_mode((win_width, win_height))

# Set up the circle
circle_radius = 10
circle_color = BLACK

# Set up the animation loop
frames = []
for i in range(num_frames):

    # Get the current time
    t = i / FPS

    # Find the closest time step to the current time
    idx = np.argmin(np.abs(np.array(time_steps) - t))
    x, y = positions[idx]

    # Clear the screen
    win.fill((255, 255, 255))

    # Draw the circle
    pygame.draw.circle(win, circle_color, (int(x), int(y)), circle_radius)

    # Update the display
    pygame.display.update()

    # Capture the frame as a NumPy array
    frame = np.array(pygame.surfarray.array3d(win))

    # Add the frame to the list of frames
    frames.append(frame)

    # Wait for the next frame
    clock.tick(FPS)

# Save the frames as a video using imageio
imageio.mimwrite(video_file, frames, fps=fps)

# Quit Pygame
pygame.quit()
