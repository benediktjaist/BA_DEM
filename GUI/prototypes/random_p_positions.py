import pygame
import random


def generate_random_circles():
    # Initialize Pygame
    pygame.init()
    screen_width = 1328
    screen_height = 832
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Random circles without intersections')

    # Define circle properties
    min_radius = 40
    max_radius = 70
    min_distance = 1 * max_radius  # Minimum distance between circles
    max_circles = 10  # Maximum number of circles

    # Define rectangle properties
    rect_top = 50
    rect_left = 50
    rect_bottom = 782
    rect_right = 1228

    # Create progress tracker message surface
    font = pygame.font.SysFont('Arial', 20)
    tracker_surface = font.render(f'Current: 0 / Desired: {max_circles}', True, (0, 0, 0))
    tracker_rect = tracker_surface.get_rect(topleft=(10, 10))

    # Create start button
    button_width = 80
    button_height = 30
    button_surface = pygame.Surface((button_width, button_height))
    button_surface.fill((0, 255, 0))
    font = pygame.font.SysFont('Arial', 16)
    text_surface = font.render('Start', True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=(button_width // 2, button_height // 2))
    button_surface.blit(text_surface, text_rect)
    button_rect = button_surface.get_rect(topright=(screen_width - 10, 10))

    # Generate circles
    circles = []
    current_circles = 0
    start_generation = False

    # Draw start button
    screen.blit(button_surface, button_rect)
    pygame.display.flip()

    while not start_generation:
        # Check for button click
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos):
                    start_generation = True

    while current_circles < max_circles:
        x = random.randint(rect_left + max_radius, rect_right - max_radius)
        y = random.randint(rect_top + max_radius, rect_bottom - max_radius)
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
            current_circles += 1

            # Update progress tracker message
            tracker_surface = font.render(f'Current: {current_circles} / Desired: {max_circles}', True, (0, 0, 0))

        # Draw circles and progress tracker message
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 0), (rect_left, rect_top, rect_right - rect_left, rect_bottom - rect_top), 2)
        for circle in circles:
            pygame.draw.circle(screen, (0, 0, 255), (circle[0], circle[1]), circle[2])
        screen.blit(tracker_surface, tracker_rect)

        pygame.display.flip()

    # Display final message
    message = font.render('Maximum number of circles reached', True, (255, 0, 0))
    message_rect = message.get_rect(centerx=screen_width // 2, centery=tracker_rect.centery)
    screen.blit(message, message_rect)
    pygame.display.flip()

    # Wait for user to close window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

if __name__ == "__main__":
    generate_random_circles()
