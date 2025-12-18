import pygame

from src.types import Vec3
from src.camera import Camera
from src.entity import Entity


filename = "example/stairs_modular_right.obj"

entity = Entity(filename, Vec3.one() * 50)

pygame.init()

SCREEN_SIZE = (800, 600)
screen = pygame.display.set_mode(SCREEN_SIZE)

camera = Camera(SCREEN_SIZE)

player_rect = pygame.Rect(400, 300, 40, 40)
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    delta_time = clock.tick(60) / 1000

    camera.fill((0, 0, 0))

    entity.draw(camera)

    entity.scale += 0.1
    entity.position += (1, 0, 0)

    screen.blit(camera, (0, 0))
    pygame.display.flip()

pygame.quit()
