import pygame
import random
from Particle import Particle
import numpy as np


def generate_random_circles(recieved_props: []):
    max_number_of_particles = recieved_props[0]
    dist_factor = recieved_props[1]
    range_radius = recieved_props[2]

    range_x_vel = recieved_props[3] #ranges sind tupel mit (min_value, max_value)
    range_y_vel = recieved_props[4]
    range_rot_vel = recieved_props[5]
    range_rot = recieved_props[6]
    range_mass = recieved_props[7]
    range_young = recieved_props[8]
    range_poisson = recieved_props[9]


    # Initialize Pygame
    pygame.init()
    screen_width = 1328
    screen_height = 832
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.SCALED)
    pygame.display.set_caption('Random particles without intersections')

    # Define rectangle properties
    rect_top = 50
    rect_left = 50
    rect_bottom = 782
    rect_right = 1228  # hardcoded also in Main_new -> receive_particle_prop_limits

    # Define particle properties
    min_radius = range_radius[0]
    max_radius = range_radius[1]
    min_distance = dist_factor * max_radius  # Minimum distance between circles
    max_circles = max_number_of_particles  # Maximum number of circles

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

    # Generate particles
    particles = []
    current_particles = 0
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

    while current_particles < max_number_of_particles:
        x = random.randint(rect_left + max_radius, rect_right - max_radius)
        y = random.randint(rect_top + max_radius, rect_bottom - max_radius)

       # random.uniform
        new_radius = random.randint(min_radius, max_radius)

        new_x_vel = random.randint(range_x_vel[0], range_x_vel[1])
        new_y_vel = random.randint(range_y_vel[0], range_y_vel[1])
        new_rot_vel = random.randint(range_rot_vel[0], range_rot_vel[1])
        new_rot = random.randint(range_rot[0], range_rot[1])
        new_mass = random.randint(range_mass[0], range_mass[1])
        new_young = random.randint(range_young[0], range_young[1])
        new_poisson = random.randint(range_poisson[0], range_poisson[1])


        # Check for intersections with all previously generated particles
        intersect = False
        for particle in particles:
            distance = ((x - particle.position[0]) ** 2 + (y - particle.position[1]) ** 2) ** 0.5 # part position x, part position y
            if distance < particle.radius + new_radius + min_distance:
                intersect = True
                break

        # Add particle if no intersections
        if not intersect:
            particles.append(
            Particle(position=np.array([x, y, 0]), velocity=np.array([new_x_vel, new_y_vel, 0]), acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, new_rot]),
                     rotation_vel=np.array([0, 0, new_rot_vel]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=new_radius, k_n=new_young, poisson=new_poisson, mass=new_mass,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])))
            current_particles += 1

            # Update progress tracker message
            tracker_surface = font.render(f'Current: {current_particles} / Desired: {max_number_of_particles}', True, (0, 0, 0))

        # Draw particles and progress tracker message
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 0), (rect_left, rect_top, rect_right - rect_left, rect_bottom - rect_top), 2)
        for particle in particles:
            pygame.draw.circle(screen, (0, 0, 255), (particle.position[0], particle.position[1]), particle.radius)
        screen.blit(tracker_surface, tracker_rect)

        pygame.display.flip()

    # Display final message
    message = font.render('Maximum number of particles reached', True, (255, 0, 0))
    message_rect = message.get_rect(centerx=screen_width // 2, centery=tracker_rect.centery)
    screen.blit(message, message_rect)
    pygame.display.flip()

    # Wait for user to close window
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    pygame.quit()

    return particles



#if __name__ == "__main__":
   #generate_random_circles()
'''    # Wait for user to close window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
'''