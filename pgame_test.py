import pygame

pygame.font.init()
pygame.font.get_fonts()

poss_fonts = ['arial', 'arialblack']
score_value = 50
font = pygame.font.SysFont('arial', 70)
# -- defs --
def show_score(x,y):
    score = font.render("Score: " + str(score_value), True, (0,0,0))