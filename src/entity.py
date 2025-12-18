from typing import List, Union

from src.types import Vec3
from src.polygon import Polygon
from src.camera import Camera


class Entity:
    def __init__(self, filename: str, position: Union[tuple, Vec3] = Vec3.zero(), scale: float = 1):
        self._load_obj(filename)
        self.position = position
        self.scale = scale


    def _load_obj(self, filename):
        self.vertices: List[Vec3] = []
        self.polygons: List[Polygon] = []
        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    self.vertices.append(
                        Vec3(
                            *list(map(lambda x: float(x), line.split()[1:]))
                        )
                    )
                elif line.startswith("f "):
                    self.polygons.append([int(v.split("/")[0]) - 1 for v in line.split()[1:]])

        for i in range(len(self.polygons)):
            self.polygons[i] = Polygon([self.vertices[j] for j in self.polygons[i]])
        self._find_position()

    def _find_position(self):
        x = [v.x for v in self.vertices]
        y = [v.y for v in self.vertices]
        z = [v.z for v in self.vertices]
        self._position = Vec3(
            x=(min(x) + max(x)) / 2,
            y=(min(y) + max(y)) / 2,
            z=(min(z) + max(z)) / 2
        )
        self._scale = 1

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        for vertice in self.vertices:
            vertice.from_vec(
                self.position + (vertice - self.position) / self.scale * value
            )
        self._scale = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: Union[tuple, Vec3]):
        diff = value - self.position
        for vertice in self.vertices:
            vertice.from_vec(vertice + diff)
        self._position = value

    def draw(self, camera: Camera):
        for polygon in self.polygons:
            polygon.draw(camera)
