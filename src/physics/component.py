from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.engine3d.object3d import Object3D


class Component:
    """Base for attachable components like Collider, Rigidbody."""

    def __init__(self):
        self.object3d: Optional['Object3D'] = None

    def on_attach(self):
        pass


class Rigidbody(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False):
        super().__init__()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0

    def add_force(self, force):
        """Simple force application."""
        self.velocity += np.array(force, dtype=np.float32) / self.mass
