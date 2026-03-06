"""
Camera3D - First-person or orbit camera for 3D scenes.
"""
import numpy as np
import math
from typing import Tuple, TYPE_CHECKING

from .component import Component

if TYPE_CHECKING:
    from .gameobject import GameObject


class Camera3D(Component):
    """
    3D Camera component.
    
    Position and rotation are controlled by the GameObject's transform.
    """
    
    def __init__(self, 
                 fov: float = 60.0,
                 near: float = 0.1,
                 far: float = 1000.0):
        """
        Initialize camera.
        
        Args:
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
        """
        super().__init__()
        self.fov = fov
        self.near = near
        self.far = far
        
        # Frustum cache
        self._frustum_cache = {
            "aspect": None,
            "fov": None,
            "tan_x": None,
            "tan_y": None,
        }
        self._target_cache = np.array([0, 0, 0], dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        """Get camera position from transform."""
        if self.game_object:
            return self.game_object.transform.position
        return np.zeros(3, dtype=np.float32)
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set camera position via transform."""
        if self.game_object:
            self.game_object.transform.position = value
    
    @property
    def target(self) -> np.ndarray:
        """Get camera target (look at)."""
        return self._target_cache
    
    @target.setter
    def target(self, value: Tuple[float, float, float]):
        """Set camera target (look at)."""
        self.look_at(value)

    @property
    def forward(self) -> np.ndarray:
        """Get forward vector from transform rotation."""
        if self.game_object:
            return self.game_object.transform.forward
        return np.array([0, 0, -1], dtype=np.float32)

    def look_at(self, target: Tuple[float, float, float]):
        """Point camera at a target."""
        self._target_cache = np.array(target, dtype=np.float32)
        if self.game_object:
            self.game_object.transform.look_at(target)

    def orbit(self, dx: float, dy: float):
        """Orbit the camera around its target."""
        if not self.game_object:
            return
        target = self._target_cache
        pos = self.position
        offset = pos - target
        
        radius = np.linalg.norm(offset)
        if radius < 0.001:
            return
            
        yaw = math.atan2(offset[0], offset[2])
        pitch = math.asin(np.clip(offset[1] / radius, -1.0, 1.0))
        
        yaw += dx
        pitch += dy
        
        pitch = np.clip(pitch, -math.pi/2 + 0.01, math.pi/2 - 0.01)
        
        new_offset = np.array([
            radius * math.sin(yaw) * math.cos(pitch),
            radius * math.sin(pitch),
            radius * math.cos(yaw) * math.cos(pitch)
        ], dtype=np.float32)
        
        self.position = target + new_offset
        self.look_at(target)

    def zoom(self, amount: float):
        """Zoom the camera towards or away from its target."""
        if not self.game_object:
            return
        target = self._target_cache
        pos = self.position
        offset = pos - target
        
        radius = np.linalg.norm(offset)
        if radius < 0.001:
            direction = np.array([0, 0, 1], dtype=np.float32)
        else:
            direction = offset / radius
            
        new_radius = max(0.1, radius + amount)
        self.position = target + direction * new_radius
    
    # Movement methods delegated to transform
    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move camera by offset."""
        if self.game_object:
            self.game_object.transform.position += np.array([dx, dy, dz], dtype=np.float32)

    def move_forward(self, distance: float):
        """Move camera forward."""
        if self.game_object:
            self.game_object.transform.position += self.forward * distance

    def move_right(self, distance: float):
        """Move camera right."""
        if self.game_object:
            self.game_object.transform.position += self.right * distance

    def move_up(self, distance: float):
        """Move camera up."""
        if self.game_object:
            self.game_object.transform.position += self.up * distance

    @property
    def right(self) -> np.ndarray:
        """Get right vector."""
        if self.game_object:
            return self.game_object.transform.right
        return np.array([1, 0, 0], dtype=np.float32)

    @property
    def up(self) -> np.ndarray:
        """Get up vector."""
        if self.game_object:
            return self.game_object.transform.up
        return np.array([0, 1, 0], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix for rendering."""
        if self.game_object:
            # View matrix is inverse of camera's world matrix
            # But we can construct it from position and rotation
            # Or use look_at with position + forward
            
            eye = self.game_object.transform.position
            # We need to handle rotation correctly.
            # Transform.get_model_matrix gives us the camera's position and orientation in world.
            # View matrix is the inverse of that.
            
            model_matrix = self.game_object.transform.get_model_matrix()
            try:
                return np.linalg.inv(model_matrix)
            except np.linalg.LinAlgError:
                return np.eye(4, dtype=np.float32)
        return np.eye(4, dtype=np.float32)
    
    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """Get the projection matrix for rendering."""
        return self._perspective_matrix(self.fov, aspect, self.near, self.far)

    # ... (Keep other methods like frustum culling, but update them to use transform) ...

    @staticmethod
    def _perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(fov) / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, (2 * far * near) / (near - far), 0]
        ], dtype=np.float32)
