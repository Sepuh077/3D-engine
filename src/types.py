from typing import Union


class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.position = (x, y, z)

    def set(self, x, y, z):
        self.position = (x, y, z)

    def from_vec(self, value: Union[tuple, "Vec3"]):
        self.position = tuple(value)

    @staticmethod
    def zero():
        return Vec3()

    @staticmethod
    def one():
        return Vec3(1, 1, 1)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def copy(self):
        return Vec3(*self)

    def __getitem__(self, index):
        return self.position[index]

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __getitem__(self, index):
        return self.position[index]

    def __eq__(self, value: Union[tuple, "Vec3"]):
        return self.x == value[0] and self.y == value[1] and self.z == value[2]

    def __add__(self, other: Union[tuple, "Vec3"]):
        return Vec3(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other: Union[tuple, "Vec3"]):
        return Vec3(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    def __mul__(self, value: int | float):
        return Vec3(self[0] * value, self[1] * value, self[2] * value)

    def __truediv__(self, value: int | float):
        return Vec3(self[0] / value, self[1] / value, self[2] / value)

    def __radd__(self, other: Union[tuple, "Vec3"]):
        return self.__add__(other)

    def __rsub__(self, other: Union[tuple, "Vec3"]):
        return Vec3(other[0] - self[0], other[1] - self[1], other[2] - self[2])

    def __rmul__(self, value: int | float):
        return self.__mul__(value)

    def __iadd__(self, other: Union[tuple, "Vec3"]):
        self.set(self[0] + other[0], self[1] + other[1], self[2] + other[2])
        return self

    def __isub__(self, other: Union[tuple, "Vec3"]):
        self.set(self[0] - other[0], self[1] - other[1], self[2] - other[2])
        return self

    def __imul__(self, value: int | float):
        self.set(self[0] * value, self[1] * value, self[2] * value)
        return self

    def __itruediv__(self, value: int | float):
        self.set(self[0] / value, self[1] / value, self[2] / value)
        return self
