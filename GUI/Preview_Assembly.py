import pygame
import numpy as np
from Colours import get_colour_tuple as col
from Particle import Particle
from Boundary import Boundary


class Assembly:

    # def __init__(self, system)
        # self.particles = system.particles
    def __init__(self, particles, boundaries, gravity):
        self.particles = particles
        self.boundaries = boundaries
        self.grid = False
        self.coords = False
        self.gravity = gravity

    def show(self):
        # Initialize Pygame
        pygame.init()

        # Set up the window dimensions
        win_width = 1328
        win_height = 828

        # Set up the Pygame window
        win = pygame.display.set_mode((win_width, win_height), pygame.SCALED)

        # Set the window title
        pygame.display.set_caption(
            "Here you can see the initial conditions of the simulation")
        # Set the font
        font = pygame.font.Font(None, 30)

        win.fill(col('white'))

        # Set the size of the grid and the size of each grid square
        GRID_SIZE = 20
        GRID_WIDTH = win_width // GRID_SIZE
        GRID_HEIGHT = win_height // GRID_SIZE

        # Define the buttons
        button_color = col('blue')
        button_font = pygame.font.SysFont('Arial', 24)
        buttons = [{'text': 'Show grid', 'pos': (win_width - 100, win_height - 790)},
                   {'text': 'Hide grid', 'pos': (win_width - 100, win_height - 730)},
                   {'text': 'Show details', 'pos': (win_width - 100, win_height - 670)},
                   {'text': 'Hide details', 'pos': (win_width - 100, win_height - 610)}]

        # Wait for user to close the window
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    for button in buttons:
                        button_rect = button_surface.get_rect(center=button['pos'])
                        if button_rect.collidepoint(mouse_pos):
                            if button['text'] == 'Show grid':
                                self.grid = True
                            elif button['text'] == 'Hide grid':
                                self.grid = False
                            elif button['text'] == 'Show details':
                                self.coords = True
                            elif button['text'] == 'Hide details':
                                self.coords = False

            # Redraw the window
            win.fill(col('white'))
            if self.gravity:
                # Set up the font
                font = pygame.font.SysFont("Arial", 20, bold=True)
                # Add the text to the textbox surface
                textgrav = font.render(("Gravity = ON"), True, col('gold'))
                textbox_width = textgrav.get_width() + 20
                textbox_height = textgrav.get_height() + 10
                textbox = pygame.Surface((textbox_width, textbox_height))
                textbox.fill(col('white'))

                # Add a black frame around the textbox
                pygame.draw.rect(textbox, col('gold'), textbox.get_rect(), 3)

                # Add the text to the textbox surface
                textbox.blit(textgrav, (10, 5))

                # Add the textbox to the screen
                textbox_rect = textbox.get_rect()
                textbox_rect.center = (100, 30)
                win.blit(textbox, textbox_rect)

            if self.grid:
                for x in range(0, win_width, GRID_SIZE):
                    pygame.draw.line(win, col('gray'), (x, 0), (x, win_height))
                for y in range(0, win_height, GRID_SIZE):
                    pygame.draw.line(win, col('gray'), (0, y), (win_width, y))

            for button in buttons:
                button_surface = pygame.Surface((150, 50))
                button_surface.fill(button_color)
                text_surface = button_font.render(button['text'], True, col('white'))
                text_rect = text_surface.get_rect(center=(75, 25))
                button_surface.blit(text_surface, text_rect)
                button_rect = button_surface.get_rect(center=button['pos'])
                win.blit(button_surface, button_rect)

            if self.boundaries == None:
                # Set up the font
                font = pygame.font.SysFont("Arial", 48)
                # Add the text to the textbox surface
                text_no_p = font.render("There are no boundaries", True, col('red'))
                textbox_width = text_no_p.get_width()+ 20
                textbox_height = text_no_p.get_height()+ 10
                textbox = pygame.Surface((textbox_width, textbox_height))
                textbox.fill(col('gray'))

                # Add a black frame around the textbox
                pygame.draw.rect(textbox, col('red'), textbox.get_rect(), 1)

                # Add the text to the textbox surface
                textbox.blit(text_no_p, (10, 5))


                # Add the textbox to the screen
                textbox_rect = textbox.get_rect()
                textbox_rect.center = (win.get_rect().centerx / 2, win.get_rect().centery/2)
                win.blit(textbox, textbox_rect)

            else:
                # Draw the boundaries
                for n_boundary in self.boundaries:
                    pygame.draw.line(win, col('black'), n_boundary.start_point, n_boundary.end_point, 5)
                    if self.coords:
                        # Set up the font
                        font = pygame.font.SysFont("Arial", 14)
                        # Add the text to the textbox surface
                        text1 = font.render("start point: " + str(n_boundary.start_point), True, col('blue'))
                        text2 = font.render("end point: " + str(n_boundary.end_point), True, col('blue'))
                        textbox_width = max(text1.get_width(), text2.get_width()) + 20
                        textbox_height = text1.get_height() + text2.get_height() + 10
                        textbox = pygame.Surface((textbox_width, textbox_height))
                        textbox.fill(col('green'))

                        # Add a black frame around the textbox
                        pygame.draw.rect(textbox, col('black'), textbox.get_rect(), 1)

                        # Add the text to the textbox surface
                        textbox.blit(text1, (10, 5))
                        textbox.blit(text2, (10, text1.get_height() + 5))

                        # Add the textbox to the screen
                        textbox_rect = textbox.get_rect()
                        textbox_rect.center = (n_boundary.start_point[0], n_boundary.start_point[1])
                        win.blit(textbox, textbox_rect)

            if self.particles == None:
                # Set up the font
                font = pygame.font.SysFont("Arial", 48)
                # Add the text to the textbox surface
                text_no_p = font.render("There are no particles", True, col('red'))
                textbox_width = text_no_p.get_width()+ 20
                textbox_height = text_no_p.get_height()+ 10
                textbox = pygame.Surface((textbox_width, textbox_height))
                textbox.fill(col('gray'))

                # Add a black frame around the textbox
                pygame.draw.rect(textbox, col('red'), textbox.get_rect(), 1)

                # Add the text to the textbox surface
                textbox.blit(text_no_p, (10, 5))


                # Add the textbox to the screen
                textbox_rect = textbox.get_rect()
                textbox_rect.center = (win.get_rect().centerx / 2, win.get_rect().centery / 2 - 100)
                win.blit(textbox, textbox_rect)

            else:

                # Draw the particles
                for n_particle in self.particles:
                    # particle position + radius
                    pygame.draw.circle(win, col('black'), (n_particle.position[0], n_particle.position[1]),
                                       n_particle.radius)
                    # particle orientation (rotation)
                    pygame.draw.line(win, (139, 0, 0),
                                     (n_particle.position[0], n_particle.position[1]),
                                     (n_particle.position[0] + n_particle.radius * np.cos(
                                         n_particle.rotation[2]),
                                      n_particle.position[1] + n_particle.radius * np.sin(
                                          n_particle.rotation[2])), 3)
                    if self.gravity:
                        pygame.draw.line(win, col('gold'), (n_particle.position[0], n_particle.position[1]),
                                         (n_particle.position[0], n_particle.position[1]+50), width=3)
                        pygame.draw.line(win, col('gold'), (n_particle.position[0], n_particle.position[1] + 50),
                                         (n_particle.position[0]+10, n_particle.position[1] + 40), width=3)
                        pygame.draw.line(win, col('gold'), (n_particle.position[0], n_particle.position[1] + 50),
                                         (n_particle.position[0]-10, n_particle.position[1] + 40), width=3)

                    if self.coords:
                        # Set up the font
                        font = pygame.font.SysFont("Arial", 14)

                        # Add the text to the textbox surface
                        text1 = font.render("position: "+str(n_particle.position), True, col('blue'))
                        text2 = font.render("velocity: " + str(n_particle.velocity), True, col('blue'))
                        text3 = font.render("radius: "+str(n_particle.radius), True, col('blue'))
                        textbox_width = max(text1.get_width(), text2.get_width(), text3.get_width()) + 20
                        textbox_height = text1.get_height() + text2.get_height() + text3.get_height() + 10
                        textbox = pygame.Surface((textbox_width, textbox_height))
                        textbox.fill(col('green'))

                        # Add a black frame around the textbox
                        pygame.draw.rect(textbox, col('black'), textbox.get_rect(), 1)

                        # Add the text to the textbox surface
                        textbox.blit(text1, (10, 5))
                        textbox.blit(text2, (10, text1.get_height() + 5))
                        textbox.blit(text3, (10, text1.get_height() + text2.get_height() + 5))

                        # Add the textbox to the screen
                        textbox_rect = textbox.get_rect()
                        textbox_rect.center = (n_particle.position[0], n_particle.position[1])
                        win.blit(textbox, textbox_rect)
                    else:
                        pass

                    # particle velocity
                    if np.linalg.norm(n_particle.velocity) != 0:
                        start_point = n_particle.position
                        end_point = n_particle.position + n_particle.velocity
                        pygame.draw.aaline(win, col('red'), (start_point[0], start_point[1]), (end_point[0], end_point[1]), 5)
                        # draw arrow
                        if start_point[0] == end_point[0]:
                            if start_point[1] < end_point[1]:
                                alpha = np.pi / 2
                            else:
                                alpha = -np.pi / 2
                        else:
                            alpha = np.arctan((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))

                        if start_point[0] <= end_point[0]:

                            pygame.draw.aaline(win, col('red'), (end_point[0], end_point[1]), (
                                end_point[0] - 20 * np.cos(alpha + np.pi / 4),
                                end_point[1] - 20 * np.sin(alpha + np.pi / 4)), 5)
                            pygame.draw.aaline(win, col('red'), (end_point[0], end_point[1]), (
                                end_point[0] - 20 * np.sin(alpha + np.pi / 4),
                                end_point[1] + 20 * np.cos(alpha + np.pi / 4)), 5)
                        else:

                            pygame.draw.aaline(win, col('red'), (end_point[0], end_point[1]), (
                                end_point[0] + 20 * np.cos(alpha + np.pi / 4),
                                end_point[1] + 20 * np.sin(alpha + np.pi / 4)), 5)
                            pygame.draw.aaline(win, col('red'), (end_point[0], end_point[1]), (
                                end_point[0] + 20 * np.sin(alpha + np.pi / 4),
                                end_point[1] - 20 * np.cos(alpha + np.pi / 4)), 5)
                    else:
                        pass

            pygame.display.update()

        # Quit Pygame
        pygame.quit()
