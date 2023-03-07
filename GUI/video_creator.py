import pygame
import os
import numpy as np
import imageio


class VideoCreator:
    def __init__(self, particles, boundaries, dt, simtime, fps=50, video_dir="C:/Users/Jaist/Desktop/ba_videos"):
        self.particles = particles
        self.boundaries = boundaries
        self.fps = fps
        self.video_dir = video_dir
        self.dt = dt
        self.simtime = simtime
        self.video_name = f"3p_box_dt{dt}_fps{fps}.mp4"
        self.time_steps = np.arange(0, self.simtime, self.dt)


    def animate(self):
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

        # Calculate the number of frames needed for the desired duration
        duration = self.time_steps[-1] - self.time_steps[0] + self.dt
        num_video_frames = int(duration * fps)
        # print('n-frames needed ', num_video_frames)
        # Calculate the number of frames computed
        num_all_frames = len(self.time_steps)

        # Set up the video writer
        video_dir = "C:/Users/Jaist/Desktop/ba_videos"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        video_file = os.path.join(video_dir, self.video_name)

        # Delete the file if it already exists
        if os.path.isfile(video_file):
            os.remove(video_file)
        # Set up the Pygame window
        win = pygame.display.set_mode((win_width, win_height), pygame.HIDDEN)

        # Set up the circle
        circle_radius = 10
        circle_color = BLACK
        # Set up the video writer with FFmpeg format
        video_writer = imageio.get_writer(video_file, fps=fps, codec='libx264',
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

        for t in range(len(video_frame_indices)):

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
            clock.tick(fps)

        # Close the video writer
        video_writer.close()

        # Quit Pygame
        pygame.quit()



