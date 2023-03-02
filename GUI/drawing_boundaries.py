import pygame
from Boundary import Boundary
from Colours import get_colour_tuple


def boundary_creator(screen):
    # Set the width and height of the screen [width, height]
    WIDTH = 700
    HEIGHT = 500

    # Set the size of the grid and the size of each grid square
    GRID_SIZE = 20
    GRID_WIDTH = WIDTH // GRID_SIZE
    GRID_HEIGHT = HEIGHT // GRID_SIZE

    # Initialize Pygame
    pygame.init()

    # Set the screen size
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Set the window title
    pygame.display.set_caption("Draw a line on the grid with your mouse to create a BOUNDARY // to start click anywhere")

    # Set the font
    font = pygame.font.Font(None, 30)

    # Create a clock to control the frame rate
    clock = pygame.time.Clock()

    # Initialize the starting and ending points for the line
    start_pos = None
    end_pos = None

    # Initialize the list of line segments
    lines = []

    # Initialize the list of boundaries
    boundaries = []

    # Set up the game loop
    done = False

    while not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If the left mouse button is pressed, snap the starting position to the nearest node of the grid
                if event.button == 1:
                    x, y = event.pos
                    start_pos = (GRID_SIZE * round(x / GRID_SIZE), GRID_SIZE * round(y / GRID_SIZE))
                    end_pos = (GRID_SIZE * round(x / GRID_SIZE), GRID_SIZE * round(y / GRID_SIZE))
            elif event.type == pygame.MOUSEMOTION:
                # If the left mouse button is pressed, update the ending position and show the line
                if event.buttons[0] == 1:
                    x, y = event.pos
                    end_pos = (GRID_SIZE * round(x / GRID_SIZE), GRID_SIZE * round(y / GRID_SIZE))

                    # Clear the screen and draw the lines and boundaries
                    screen.fill(get_colour_tuple('white'))
                    for line in lines:
                        pygame.draw.line(screen, get_colour_tuple('red'), line[0], line[1], 5)
                    for boundary in boundaries:
                        pygame.draw.line(screen, get_colour_tuple('green'), boundary.start_point, boundary.end_point, 5)
                    # Draw the current line being drawn
                    pygame.draw.line(screen, get_colour_tuple('red'), start_pos, end_pos, 5)

                    # Update the screen
                    pygame.display.flip()


            elif event.type == pygame.MOUSEBUTTONUP:
                # If the left mouse button is released, add the line to the list of lines
                if event.button == 1:
                    # Check that the starting and ending positions are different
                    if start_pos != end_pos:
                        lines.append((start_pos, end_pos))
                        # Draw the red line
                        pygame.draw.line(screen, get_colour_tuple('red'), start_pos, end_pos, 5)
                        # Create a boundary with the two points of the line and add it to the list of boundaries
                        boundaries.append(Boundary(start_pos, end_pos))
                    # Clear the list of lines
                    lines = []

                    # Create a boundary with the two points of the line and add it to the list of boundaries
                    if start_pos != end_pos:
                        boundaries.append(Boundary(start_pos, end_pos))

                    # Clear the screen and draw the boundaries
                    screen.fill(get_colour_tuple('white'))
                    for boundary in boundaries:
                        pygame.draw.line(screen, get_colour_tuple('green'), boundary.start_point, boundary.end_point, 5)

                    # Update the screen
                    pygame.display.flip()

        # Draw the grid as nodes
        for x in range(0, WIDTH, GRID_SIZE):
            for y in range(0, HEIGHT, GRID_SIZE):
                pygame.draw.circle(screen, get_colour_tuple('black'), (x, y), 2)

        # Draw the lines
        for line in lines:
            pygame.draw.line(screen, get_colour_tuple('red'), line[0], line[1], 5)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If the left mouse button is pressed, snap the starting position to the nearest node of the grid
                if event.button == 1:
                    x, y = event.pos
                    start_pos = (GRID_SIZE * round(x / GRID_SIZE), GRID_SIZE * round(y / GRID_SIZE))
                    end_pos = (GRID_SIZE * round(x / GRID_SIZE), GRID_SIZE * round(y / GRID_SIZE))
            elif event.type == pygame.MOUSEMOTION:
                # If the left mouse button is pressed, update the ending position and show the line
                if event.buttons[0] == 1:
                    x, y = event.pos
                    end_pos = (GRID_SIZE * round(x / GRID_SIZE), GRID_SIZE * round(y / GRID_SIZE))
            elif event.type == pygame.MOUSEBUTTONUP:
                # If the left mouse button is released, add the line to the list of lines
                if event.button == 1:
                    lines.append((start_pos, end_pos))
                    # Draw the red line
                    pygame.draw.line(screen, get_colour_tuple('red'), start_pos, end_pos, 5)
                    # Create a boundary with the two points of the line and add it to the list of boundaries
                    boundaries.append(Boundary(start_pos, end_pos))
                    # Clear the list of lines
                    lines = []

        # Draw the boundaries
        for boundary in boundaries:
            pygame.draw.line(screen, get_colour_tuple('green'), boundary.start_point, boundary.end_point, 5)

        # Display the current position of the mouse in steps of 10
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos_rounded = (round(mouse_pos[0] / 10) * 10, round(mouse_pos[1] / 10) * 10)
        pos_text = font.render(f"Mouse position: {mouse_pos_rounded}", True, get_colour_tuple('black'), (255, 255, 255))
        screen.blit(pos_text, (10, 30))
        # Update the screen
        pygame.display.flip()

        # Control the frame rate
        clock.tick(60)

    # Quit Pygame
    pygame.quit()
    return boundaries


# this ensures that the GUI can only be called
# here and not through an import for e.g. test purposes
if __name__ == "__main__":
    boundaries = boundary_creator()
    print(len(boundaries))
    for boundary in boundaries:
        print('start point ', boundary.start_point)
        print('end point ', boundary.end_point)
        print(type(boundary))

# sometimes there are visible artefacts (GREEN) of boundaries, this is probably due to the
# incorrect condition for the creation of boundaries only at the nodes.
