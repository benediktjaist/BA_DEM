import pygame
import os
from better_gui import particle_creator
from Particle import Particle
from dem_solver import System
from Colours import colors
import numpy as np

WIDTH, HEIGHT = 900, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)


# Create a button class
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (0, 0, 255)
        self.text_color = (255, 255, 255)
        self.text = text
        self.font = pygame.font.SysFont(None, 30)
        self.rendered_text = self.font.render(text, True, self.text_color)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        surface.blit(self.rendered_text, (self.rect.x + self.rect.width // 2 - self.rendered_text.get_width() // 2,
                                          self.rect.y + self.rect.height // 2 - self.rendered_text.get_height() // 2))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


def draw_window():
    WIN.fill(WHITE)
    # Load the image
    tum_logo = pygame.image.load('TUM_Logo_blau_rgb_p.png')
    resized_image = pygame.transform.scale(tum_logo, (200, 100))

    # Draw the image onto the screen called WIN
    WIN.blit(resized_image, (600, 100))  # the second arg is position
    create_particles_button.draw(WIN)
    start_simulation_button.draw(WIN)
    show_animation_button.draw(WIN)


def main():
    global create_particles_button, start_simulation_button, show_animation_button
    run = True
    pygame.font.init()

    # Initialize the buttons
    create_particles_button = Button(WIDTH // 2 - 100, HEIGHT // 2 - 25, 200, 50, "Create Particles")
    start_simulation_button = Button(WIDTH // 2 - 100, HEIGHT // 2 + 50, 200, 50, "Start Simulation")
    show_animation_button = Button(WIDTH // 2 - 100, HEIGHT // 2 + 125, 200, 50, "Show Animation")

    particles = []

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Check for button clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if create_particles_button.is_clicked(pos):
                    particles = particle_creator()
                elif start_simulation_button.is_clicked(pos):
                    system = System(particles=particles, dt=0.01, simtime=2, mu=0.7, coeff_of_restitution=1) # system properties
                    system.run_simulation()
                    positions = system.get_positions()
                elif show_animation_button.is_clicked(pos):
                    # Open a new Pygame window
                    animation_win = pygame.display.set_mode((WIDTH+200, HEIGHT+200))

                    # Draw particles for each time step
                    # Draw particles for each time step
                    for t in range(len(positions)):
                        animation_win.fill(WHITE)
                        for particle_idx in range(len(particles)):
                            particle_pos = positions[t][particle_idx][-1]
                            particle_radius = particles[particle_idx].radius
                            particle_rot = particles[particle_idx].rotation[2]

                            # Draw circle
                            pygame.draw.circle(animation_win, colors['black'],
                                               (int(particle_pos[0]), int(particle_pos[1])), int(particle_radius))

                            # Calculate endpoint of red line (based on particle rotation)
                            x = particle_pos[0] + particle_radius * np.cos(particle_rot)
                            y = particle_pos[1] + particle_radius * np.sin(particle_rot)
                            end_point = (int(x), int(y))

                            # Draw red line from center of particle to endpoint
                            pygame.draw.line(animation_win, colors['red'], (particle_pos[0], particle_pos[1]), end_point, 3)

                        pygame.display.update()
                        pygame.time.delay(1000)  # the second window is closed immideatly

        draw_window()
        pygame.display.update()

    pygame.quit()


# this ensures that the game can only be called
# here and not through an import
if __name__ == "__main__":
    main()
