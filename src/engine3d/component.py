from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .gameobject import GameObject

class Component:
    """Base for attachable components like Transform, Object3D, Collider, Rigidbody."""

    def __init__(self):
        self.game_object: Optional['GameObject'] = None

    def on_attach(self):
        pass

    def update(self, delta_time: float):
        pass


class Transform(Component):
    """Component storing position, rotation, and scale."""

    def __init__(self):
        super().__init__()
        self._position = np.zeros(3, dtype=np.float32)
        self._rotation = np.zeros(3, dtype=np.float32)
        self._scale = np.ones(3, dtype=np.float32)

        self._transform_dirty = True
        self._cached_model = None
        self._cached_rotation = None
        self._prev_position = np.copy(self._position)

    def _mark_dirty(self):
        self._transform_dirty = True
        if self.game_object:
            from src.physics import Collider
            for comp in self.game_object.get_components(Collider):
                comp._transform_dirty = True

    def _update_prev_position(self):
        self._prev_position = np.copy(self._position)

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @position.setter
    def position(self, value):
        self._update_prev_position()
        self._position = np.array(value, dtype=np.float32)
        self._mark_dirty()

    @property
    def x(self) -> float:
        return float(self._position[0])

    @x.setter
    def x(self, value: float):
        self.position = (value, self._position[1], self._position[2])

    @property
    def y(self) -> float:
        return float(self._position[1])

    @y.setter
    def y(self, value: float):
        self.position = (self._position[0], value, self._position[2])

    @property
    def z(self) -> float:
        return float(self._position[2])

    @z.setter
    def z(self, value: float):
        self.position = (self._position[0], self._position[1], value)

    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        self._update_prev_position()
        self._position += np.array([dx, dy, dz], dtype=np.float32)
        self._mark_dirty()

    @property
    def rotation(self) -> tuple:
        return tuple(np.degrees(self._rotation))

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.radians(value).astype(np.float32)
        self._mark_dirty()

    @property
    def rotation_x(self) -> float:
        return float(np.degrees(self._rotation[0]))

    @rotation_x.setter
    def rotation_x(self, value: float):
        self._rotation[0] = np.radians(value)
        self._mark_dirty()

    @property
    def rotation_y(self) -> float:
        return float(np.degrees(self._rotation[1]))

    @rotation_y.setter
    def rotation_y(self, value: float):
        self._rotation[1] = np.radians(value)
        self._mark_dirty()

    @property
    def rotation_z(self) -> float:
        return float(np.degrees(self._rotation[2]))

    @rotation_z.setter
    def rotation_z(self, value: float):
        self._rotation[2] = np.radians(value)
        self._mark_dirty()

    def rotate(self, dx: float = 0, dy: float = 0, dz: float = 0):
        self._rotation += np.radians([dx, dy, dz]).astype(np.float32)
        self._mark_dirty()

    @property
    def scale(self) -> float:
        return float(self._scale[0])

    @scale.setter
    def scale(self, value: float):
        self._scale = np.array([value, value, value], dtype=np.float32)
        self._mark_dirty()

    @property
    def scale_xyz(self) -> tuple:
        return tuple(self._scale)

    @scale_xyz.setter
    def scale_xyz(self, value):
        self._scale = np.array(value, dtype=np.float32)
        self._mark_dirty()

    def get_model_matrix(self) -> np.ndarray:
        if not self._transform_dirty:
            return self._cached_model

        cx, cy, cz = np.cos(self._rotation)
        sx, sy, sz = np.sin(self._rotation)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rx @ Ry @ Rz
        self._cached_rotation = R

        s_x, s_y, s_z = self._scale
        tx, ty, tz = self._position
        S = np.array([[s_x, 0, 0, 0], [0, s_y, 0, 0], [0, 0, s_z, 0], [0, 0, 0, 1]], dtype=np.float32)
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = R
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [tx, ty, tz, 1]], dtype=np.float32)
        self._cached_model = S @ R4 @ T
        self._transform_dirty = False
        return self._cached_model


class Rigidbody(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False):
        super().__init__()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static

    def add_force(self, force):
        """Simple force application."""
        self.velocity += np.array(force, dtype=np.float32) / self.mass

    def update(self, delta_time: float):
        if self.is_static or self.is_kinematic:
            return

        if self.use_gravity:
            # Simple gravity: 9.81 m/s^2 downwards
            self.velocity[1] -= 9.81 * delta_time

        # if self.game_object and np.any(self.velocity):
        #     # Apply velocity to position
        #     self.game_object.transform.move(*(self.velocity * delta_time))
