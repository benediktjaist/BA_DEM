import pygame
import random

# Initialize Pygame
pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Random circles without intersections')

# Define circle properties
min_radius = 10
max_radius = 50
min_distance = 2 * max_radius  # Minimum distance between circles

# Generate circles
circles = []
while True:
    x = random.randint(max_radius, screen_width - max_radius)
    y = random.randint(max_radius, screen_height - max_radius)
    radius = random.randint(min_radius, max_radius)

    # Check for intersections with all previously generated circles
    intersect = False
    for circle in circles:
        distance = ((x - circle[0]) ** 2 + (y - circle[1]) ** 2) ** 0.5
        if distance < circle[2] + radius + min_distance:
            intersect = True
            break

    # Add circle if no intersections
    if not intersect:
        circles.append((x, y, radius))

    # Draw circles
    screen.fill((255, 255, 255))
    for circle in circles:
        pygame.draw.circle(screen, (0, 0, 255), (circle[0], circle[1]), circle[2])
    pygame.display.flip()

    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
