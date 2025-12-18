import pygame
import math

from src.types import Vec3


class Camera(pygame.Surface):
    def __init__(self, size, position: Vec3 = Vec3.zero(), angle: float = 0, spread: float = math.pi / 6):
        super().__init__(size)
        self.position = position # Does not have any effect yet
        self.angle = angle # Does not have any effect yet
        self.spread = spread

    @property
    def left(self):
        return self.position.x

    @property
    def right(self):
        return self.left + self.get_width()

    @property
    def bottom(self):
        return self.position.y

    @property
    def top(self):
        return self.bottom + self.get_height()

    def world_to_cam(self, vec: Vec3):
        diff = vec.z * math.tan(self.spread)
        h = self.get_height() + 2 * diff
        w = self.get_width() + 2 * diff
        pos = (
            (vec.x - self.left + diff) / w * self.get_width(),
            (vec.y - self.bottom + diff) / h * self.get_height(),
        )
        return pos
