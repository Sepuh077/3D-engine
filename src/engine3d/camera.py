"""
Camera3D - First-person or orbit camera for 3D scenes.
"""
import numpy as np
import math
from typing import Tuple, Optional


class Camera3D:
    """
    3D Camera with position, target, and projection settings.
    
    Supports both first-person and orbit camera modes.
    
    Example:
        camera = Camera3D()
        camera.position = (0, 5, 10)
        camera.look_at((0, 0, 0))
    """
    
    def __init__(self, 
                 position: Tuple[float, float, float] = (0, 5, 10),
                 target: Tuple[float, float, float] = (0, 0, 0),
                 fov: float = 60.0,
                 near: float = 0.1,
                 far: float = 1000.0):
        """
        Initialize camera.
        
        Args:
            position: Camera position (x, y, z)
            target: Point the camera looks at
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
        """
        self._position = np.array(position, dtype=np.float32)
        self._target = np.array(target, dtype=np.float32)
        self._up = np.array([0, 1, 0], dtype=np.float32)
        
        self.fov = fov
        self.near = near
        self.far = far
        
        # For orbit camera
        self._orbit_distance = np.linalg.norm(self._position - self._target)
        self._orbit_yaw = 0.0
        self._orbit_pitch = 0.0
        self._update_orbit_angles()

        # Frustum cache
        self._frustum_cache = {
            "aspect": None,
            "fov": None,
            "tan_x": None,
            "tan_y": None,
        }
    
    def _update_orbit_angles(self):
        """Calculate orbit angles from current position."""
        diff = self._position - self._target
        self._orbit_distance = np.linalg.norm(diff)
        if self._orbit_distance > 0:
            self._orbit_yaw = math.atan2(diff[0], diff[2])
            self._orbit_pitch = math.asin(np.clip(diff[1] / self._orbit_distance, -1, 1))
    
    def _update_position_from_orbit(self):
        """Update position based on orbit angles."""
        x = self._orbit_distance * math.cos(self._orbit_pitch) * math.sin(self._orbit_yaw)
        y = self._orbit_distance * math.sin(self._orbit_pitch)
        z = self._orbit_distance * math.cos(self._orbit_pitch) * math.cos(self._orbit_yaw)
        self._position = self._target + np.array([x, y, z], dtype=np.float32)
    
    @property
    def position(self) -> np.ndarray:
        """Get camera position."""
        return self._position.copy()
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set camera position."""
        self._position = np.array(value, dtype=np.float32)
        self._update_orbit_angles()
    
    @property
    def target(self) -> np.ndarray:
        """Get camera target (look-at point)."""
        return self._target.copy()
    
    @target.setter
    def target(self, value: Tuple[float, float, float]):
        """Set camera target."""
        self._target = np.array(value, dtype=np.float32)
        self._update_orbit_angles()
    
    @property
    def x(self) -> float:
        return float(self._position[0])
    
    @x.setter
    def x(self, value: float):
        self._position[0] = value
        self._update_orbit_angles()
    
    @property
    def y(self) -> float:
        return float(self._position[1])
    
    @y.setter
    def y(self, value: float):
        self._position[1] = value
        self._update_orbit_angles()
    
    @property
    def z(self) -> float:
        return float(self._position[2])
    
    @z.setter
    def z(self, value: float):
        self._position[2] = value
        self._update_orbit_angles()
    
    def look_at(self, target: Tuple[float, float, float]):
        """Point camera at a target."""
        self._target = np.array(target, dtype=np.float32)
        self._update_orbit_angles()
    
    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move camera by offset."""
        self._position += np.array([dx, dy, dz], dtype=np.float32)
        self._target += np.array([dx, dy, dz], dtype=np.float32)
    
    def move_forward(self, distance: float):
        """Move camera forward (towards target)."""
        direction = self._target - self._position
        direction = direction / np.linalg.norm(direction)
        self._position += direction * distance
        self._target += direction * distance
    
    def move_right(self, distance: float):
        """Move camera to the right."""
        forward = self._target - self._position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self._up)
        right = right / np.linalg.norm(right)
        self._position += right * distance
        self._target += right * distance
    
    def move_up(self, distance: float):
        """Move camera up."""
        self._position[1] += distance
        self._target[1] += distance
    
    # Orbit camera controls
    def orbit(self, yaw_delta: float = 0, pitch_delta: float = 0):
        """
        Orbit camera around target.
        
        Args:
            yaw_delta: Horizontal rotation in radians
            pitch_delta: Vertical rotation in radians
        """
        self._orbit_yaw += yaw_delta
        self._orbit_pitch += pitch_delta
        
        # Clamp pitch to avoid flipping
        self._orbit_pitch = np.clip(self._orbit_pitch, -math.pi/2 + 0.1, math.pi/2 - 0.1)
        
        self._update_position_from_orbit()
    
    def zoom(self, delta: float):
        """
        Zoom in/out (change distance to target).
        
        Args:
            delta: Positive to zoom out, negative to zoom in
        """
        self._orbit_distance = max(1.0, self._orbit_distance + delta)
        self._update_position_from_orbit()
    
    @property
    def distance(self) -> float:
        """Get distance from camera to target."""
        return float(self._orbit_distance)
    
    @distance.setter
    def distance(self, value: float):
        """Set distance from camera to target."""
        self._orbit_distance = max(0.1, value)
        self._update_position_from_orbit()
    
    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix for rendering."""
        return self._look_at_matrix(self._position, self._target, self._up)
    
    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """Get the projection matrix for rendering."""
        return self._perspective_matrix(self.fov, aspect, self.near, self.far)

    def world_to_view(self, point: np.ndarray) -> np.ndarray:
        """Transform a world-space point to view space."""
        p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        return p @ self.get_view_matrix()

    def is_sphere_visible(self, center: np.ndarray, radius: float, aspect: float) -> bool:
        """
        Quick frustum check for a bounding sphere.
        Uses the camera's fov/near/far and row-major view transform.
        """
        view = self.get_view_matrix()
        tan_x, tan_y = self.get_frustum_tangents(aspect)
        return self.is_sphere_visible_view(center, radius, view, aspect, tan_x, tan_y)

    def get_frustum_tangents(self, aspect: float) -> Tuple[float, float]:
        if (self._frustum_cache["aspect"] != aspect or
                self._frustum_cache["fov"] != self.fov):
            tan_y = math.tan(math.radians(self.fov) * 0.5)
            tan_x = tan_y * aspect
            self._frustum_cache["aspect"] = aspect
            self._frustum_cache["fov"] = self.fov
            self._frustum_cache["tan_x"] = tan_x
            self._frustum_cache["tan_y"] = tan_y
        return self._frustum_cache["tan_x"], self._frustum_cache["tan_y"]

    def is_sphere_visible_view(self, center: np.ndarray, radius: float,
                               view: np.ndarray, aspect: float,
                               tan_x: Optional[float] = None, tan_y: Optional[float] = None) -> bool:
        if tan_x is None or tan_y is None:
            tan_x, tan_y = self.get_frustum_tangents(aspect)

        p = np.array([center[0], center[1], center[2], 1.0], dtype=np.float32)
        c = p @ view
        z = c[2]

        # In view space, camera looks down -Z.
        if z > -self.near + radius:
            return False
        if z < -self.far - radius:
            return False

        max_x = -z * tan_x + radius
        max_y = -z * tan_y + radius

        if abs(c[0]) > max_x:
            return False
        if abs(c[1]) > max_y:
            return False

        return True
    
    @staticmethod
    def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create view matrix using look-at algorithm."""
        f = target - eye
        f = f / np.linalg.norm(f)
        
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)
        
        u = np.cross(r, f)
        
        rotation = np.array([
            [r[0], u[0], -f[0], 0],
            [r[1], u[1], -f[1], 0],
            [r[2], u[2], -f[2], 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        translation = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-eye[0], -eye[1], -eye[2], 1]
        ], dtype=np.float32)
        
        return translation @ rotation
    
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
