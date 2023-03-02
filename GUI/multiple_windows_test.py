import pygame

# Initialize Pygame
pygame.init()

# Set up the main window
main_window_size = (400, 400)
main_window = pygame.display.set_mode(main_window_size)
pygame.display.set_caption("Main Window")

# Set up the second window
second_window_size = (200, 200)
second_window = pygame.Surface(second_window_size)
second_window.fill((255, 255, 255))

# Set up the back button
back_button_size = (50, 30)
back_button_pos = (10, 10)
back_button_rect = pygame.Rect(back_button_pos, back_button_size)
back_button_surf = pygame.Surface(back_button_size)
back_button_surf.fill((255, 0, 0))
font = pygame.font.Font(None, 20)
back_button_text = font.render("Back", True, (255, 255, 255))
back_button_text_pos = (back_button_size[0] // 2 - back_button_text.get_width() // 2,
                        back_button_size[1] // 2 - back_button_text.get_height() // 2)
back_button_surf.blit(back_button_text, back_button_text_pos)

# Set up the state variable to keep track of which window is active
active_window = "main"

# Run the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if active_window == "main" and back_button_rect.collidepoint(event.pos):
                active_window = "second"
            elif active_window == "second" and back_button_rect.collidepoint(event.pos):
                active_window = "main"

    # Update the display
    if active_window == "main":
        main_window.fill((0, 0, 255))
        pygame.draw.rect(main_window, (255, 0, 0), back_button_rect)
        main_window.blit(back_button_surf, back_button_pos)
        pygame.display.flip()
    elif active_window == "second":
        second_window.fill((255, 255, 255))
        pygame.draw.rect(second_window, (255, 0, 0), back_button_rect)
        second_window.blit(back_button_surf, back_button_pos)
        main_window.blit(second_window, (100, 100))
        pygame.display.update((100, 100, second_window_size[0], second_window_size[1]))

# Quit Pygame
pygame.quit()
