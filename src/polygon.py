import pygame
from typing import List

from .types import Vec3
from .camera import Camera


class Polygon:
    def __init__(self, vecs: List[Vec3]):
        self.vecs = vecs

    def is_visible(self, points: List[tuple], camera: Camera):
        for point in points:
            if camera.left <= point[0] <= camera.right and \
                    camera.bottom <= point[1] <= camera.top:
                return True
        return False

    def get_points(self, camera: Camera):
        points = []
        for vec in self.vecs:
            points.append(
                camera.world_to_cam(vec)
            )
        return self.is_visible(points, camera), points

    def draw(self, camera: Camera):
        visible, points = self.get_points(camera)
        if visible:
            pygame.draw.polygon(camera, (255, 0, 0), points)
