import pygame
import os
import numpy as np
import imageio

# Initialize Pygame
pygame.init()

# Set up the window dimensions
win_width = 1280
win_height = 720

# Set up the frame rate and clock
fps = 50
clock = pygame.time.Clock()

# Set up the colors
BLACK = (0, 0, 0)

# Set up the positions of the circle center
# positions = [(0, 100), (2, 100), (4, 100), (6, 100), (8, 100)]

# Set up the time steps
# time_steps = [0, 0.5, 1, 1.5, 2]

positions = []
time_steps = []
# v = s/t -> s = v*t
v = 20
dt = 0.01

for x in np.arange(0, 2, dt):
    time_steps.append(round(x, 10))
    x = x*v
    positions.append((round(x+100, 10), 100))

# Calculate the number of frames needed for the desired duration
duration = time_steps[-1] - time_steps[0]+dt
num_video_frames = int(duration * fps)

# Calculate the number of frames computed
num_all_frames = len(time_steps)


# Set up the video writer
video_dir = "C:/Users/Jaist/Desktop/ba_videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
video_file = os.path.join(video_dir, f"animation_dt{dt}_fps{fps}.mp4")

# Delete the file if it already exists
if os.path.isfile(video_file):
    os.remove(video_file)
# Set up the Pygame window
win = pygame.display.set_mode((win_width, win_height), pygame.HIDDEN)

# Set up the circle
circle_radius = 10
circle_color = BLACK
# Set up the video writer with FFmpeg format
#video_writer = imageio.get_writer(video_file, fps=fps, codec='libx264',
                                  #ffmpeg_params=['-pix_fmt', 'yuv420p', '-vf', 'transpose=1,hflip',
                                                 #'-ip2:v', '10M', '-preset', 'slow', '-probesize', '1000M', '-analyzeduration', '1000M'])
video_writer = imageio.get_writer(video_file, fps=fps, codec='libx264',
                                  ffmpeg_params=['-pix_fmt', 'yuv420p', '-vf', 'transpose=1,hflip',
                                                  '-preset', 'slow', '-probesize', '1000M', '-analyzeduration', '1000M'])


# animation loop
# frame step size: if frame step size is e.g. 3, only every third frame is used for the video
frame_step_size = num_all_frames / num_video_frames

# saves the positions needed for the video
video_frame_indices = [i for i in range(num_all_frames) if i % frame_step_size == 0]

for i in video_frame_indices:
    x, y = positions[i]
    # Clear the screen
    win.fill((255, 255, 255))

    # Draw the circle
    pygame.draw.circle(win, circle_color, (int(x), int(y)), circle_radius)
    pygame.draw.line(win, BLACK, (300, 500), (400, 500), 5)

    # Update the display
    pygame.display.update()

    # Capture the frame as ip1 NumPy array
    frame = np.array(pygame.surfarray.array3d(win))

    # Add the frame to the video writer
    video_writer.append_data(frame)

    # Wait for the next frame
    clock.tick(fps)

# Close the video writer
video_writer.close()

# Quit Pygame
pygame.quit()

'''
# Set up the animation loop
for i in range(num_video_frames):
    # Get the current time
    t = i / FPS

    # Find the closest time step to the current time
    idx = np.argmin(np.abs(np.array(time_steps) - t))
    x, y = positions[idx]
'''
print(video_frame_indices)
print(len(video_frame_indices))


