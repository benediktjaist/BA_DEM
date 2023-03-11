import pygame
import os
import numpy as np
import imageio
import time
from PyQt6.QtCore import QObject, pyqtSignal


class VideoCreator(QObject):
    vid_iterationChanged = pyqtSignal(int)
    vid_total_iterationsChanged = pyqtSignal(int)
    vid_remaining_timeChanged = pyqtSignal(float)

    def __init__(self, particles, boundaries, dt, simtime, video_name, fps=50, video_dir="C:/Users/Jaist/Desktop/ba_videos"):
        super().__init__()
        self.particles = particles
        self.boundaries = boundaries
        self.fps = fps
        self.video_dir = video_dir
        self.dt = dt
        self.simtime = simtime
        self.video_name = video_name # "aaa.mp4" # f"aa{len(particles)}p_{len(boundaries)}b_{dt}_fps{fps}.mp4"
        self.time_steps = np.arange(0, self.simtime, self.dt)
        self.total_iterations = int(simtime/dt)
        self.elapsed_time = 0
        self.remaining_time = 0


    def animate(self):
        start_time = time.time()
        # Initialize Pygame
        pygame.init()

        # Set up the window dimensions
        win_width = 1328
        win_height = 832

        # Set up the frame rate and clock
        clock = pygame.time.Clock()

        # Set up the colors
        BLACK = (0, 0, 0)

        # Calculate the number of frames needed for the desired duration
        duration = self.time_steps[-1] - self.time_steps[0] + self.dt
        num_video_frames = int(duration * self.fps)
        # print('n-frames needed ', num_video_frames)
        # Calculate the number of frames computed
        num_all_frames = len(self.time_steps)

        # Set up the video writer
        # video_dir = "C:/Users/Jaist/Desktop/ba_videos"
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        video_file = os.path.join(self.video_dir, self.video_name)
        # video_file = os.path.normpath(os.path.join(self.video_dir, self.video_name))
        #video_file = os.path.join(self.video_dir.rstrip('\\'), self.video_name)
        #video_file = os.path.normpath(
            #os.path.join(self.video_dir.replace('\\', '/'), self.video_name.replace('\\', '/')))

        # Delete the file if it already exists
        if os.path.isfile(video_file):
            os.remove(video_file)
        # Set up the Pygame window
        win = pygame.display.set_mode((win_width, win_height), pygame.HIDDEN)

        # Set up the circle
        circle_radius = 10
        circle_color = BLACK
        # Set up the video writer with FFmpeg format
        video_writer = imageio.get_writer(video_file, fps=self.fps, codec='libx264',
                                          ffmpeg_params=['-pix_fmt', 'yuv420p', '-vf', 'transpose=1,hflip',
                                                          '-preset', 'slow', '-probesize', '1000M',
                                                         '-analyzeduration', '1000M'])

        # animation loop
        # frame step size: if frame step size is e.g. 3, only every third frame is used for the video
        frame_step_size = num_all_frames / num_video_frames

        # calculate the positions needed for the video
        video_frame_indices = [i for i in range(num_all_frames) if i % frame_step_size == 0]
        # print('vframeindices ',len(video_frame_indices))
        # print(video_frame_indices)
        # crop the historic variables from the particles to the size needed for the video
        for particle in self.particles:
            cropped_historic_positions = [particle.historic_positions[i] for i in video_frame_indices]
            particle.historic_positions = cropped_historic_positions
            cropped_historic_rotations = [particle.historic_rotations[i] for i in video_frame_indices]
            particle.historic_rotations = cropped_historic_rotations
            # print(particle.historic_positions)
            # print(len(particle.historic_positions))

        for iteration, t in enumerate(range(len(video_frame_indices))):

            win.fill((255, 255, 255))

            # Draw the particles
            for n_particle in self.particles:
                pygame.draw.circle(win, BLACK, (n_particle.historic_positions[t][0], n_particle.historic_positions[t][1]),
                               n_particle.radius)
                pygame.draw.line(win, (139, 0, 0), (n_particle.historic_positions[t][0], n_particle.historic_positions[t][1]),
                             (n_particle.historic_positions[t][0] + n_particle.radius * np.cos(n_particle.historic_rotations[t][2]),
                              n_particle.historic_positions[t][1] + n_particle.radius * np.sin(n_particle.historic_rotations[t][2])), 3)

            # Draw the boundaries
            for n_boundary in self.boundaries:
                pygame.draw.line(win, BLACK, (n_boundary.start_point[0], n_boundary.start_point[1]),
                             (n_boundary.end_point[0], n_boundary.end_point[1]), width=5)
                # pygame.draw.circle(win, (255, 255, 255), (n_boundary.start_point[0], n_boundary.start_point[1]), radius=5)
                # pygame.draw.circle(win, (0, 0, 0), (n_boundary.end_point[0], n_boundary.end_point[1]), radius=5)

            # Update the display
            pygame.display.update()

            # Capture the frame as ip1 NumPy array
            frame = np.array(pygame.surfarray.array3d(win))

            # Add the frame to the video writer
            video_writer.append_data(frame)

            # Wait for the next frame
            clock.tick(self.fps)

            # Progress Tracker
            self.elapsed_time = time.time() - start_time
            self.remaining_time = (num_video_frames - iteration - 1) * self.elapsed_time / (iteration + 1)
            self.vid_iterationChanged.emit(iteration + 1)
            self.vid_total_iterationsChanged.emit(num_video_frames)
            self.vid_remaining_timeChanged.emit(self.remaining_time)

        # Close the video writer
        video_writer.close()

        # Quit Pygame
        pygame.quit()


'''
print('zeit', f"{video.remaining_time[0]:02d}:{video.remaining_time[1]:02d}:{video.remaining_time[2]:02d}")

# Conversion to hh:mm:ss
            elapsed_seconds = int(self.elapsed_time)
            remaining_seconds = int(remaining_time_secs)
            total_seconds = elapsed_seconds + remaining_seconds
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            print('done')
            
            
            self.remaining_time = [hours, minutes, seconds]

'''