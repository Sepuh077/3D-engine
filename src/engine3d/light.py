"""
Light3D - Lighting for 3D scenes.
"""
import numpy as np
from typing import Tuple
from .graphics.color import ColorType
from .component import Component


class Light3D(Component):
    """
    Directional light for 3D scenes.
    
    Example:
        light_go = GameObject("Light")
        light = Light3D(color=Color.WHITE)
        light_go.add_component(light)
        # set direction by rotating the GameObject
        light_go.transform.rotation = (-45, 30, 0)
    """
    
    def __init__(self, 
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 ambient: float = 0.2):
        """
        Initialize light.
        
        Args:
            color: Light color (RGB 0-1)
            intensity: Light intensity multiplier
            ambient: Ambient light level (0-1)
        """
        super().__init__()
        self._fallback_direction = np.array([0.3, -0.7, -0.5], dtype=np.float32)
        self._normalize_fallback_direction()
        
        self.color = color
        self.intensity = intensity
        self.ambient = ambient
    
    def _normalize_fallback_direction(self):
        """Normalize the fallback direction vector."""
        norm = np.linalg.norm(self._fallback_direction)
        if norm > 0:
            self._fallback_direction /= norm
    
    @property
    def direction(self) -> np.ndarray:
        """Get light direction. If attached to a GameObject, derives from transform rotation."""
        if self.game_object and self.game_object.transform:
            model = self.game_object.transform.get_model_matrix()
            # Forward vector is usually -Z in this coordinate system
            # 3rd column of the rotation matrix (index 2)
            fwd = -model[0:3, 2]
            norm = np.linalg.norm(fwd)
            if norm > 0:
                fwd /= norm
            return fwd
        return self._fallback_direction.copy()
    
    @direction.setter
    def direction(self, value: Tuple[float, float, float]):
        """Set fallback light direction."""
        self._fallback_direction = np.array(value, dtype=np.float32)
        self._normalize_fallback_direction()
    
    def point_from(self, position: Tuple[float, float, float], 
                   target: Tuple[float, float, float] = (0, 0, 0)):
        """
        Set fallback light to point from a position towards a target.
        """
        if self.game_object and self.game_object.transform:
            self.game_object.transform.position = position
        pos = np.array(position, dtype=np.float32)
        tgt = np.array(target, dtype=np.float32)
        self._fallback_direction = tgt - pos
        self._normalize_fallback_direction()


class PointLight3D(Component):
    """
    Point light that emits in all directions from a position.
    (For future implementation with more advanced shaders)
    """
    
    def __init__(self,
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 range: float = 50.0):
        """
        Initialize point light.
        
        Args:
            color: Light color (RGB 0-1)  
            intensity: Light intensity
            range: Maximum light range
        """
        super().__init__()
        self._fallback_position = np.array([0, 10, 0], dtype=np.float32)
        self.color = color
        self.intensity = intensity
        self.range = range
    
    @property
    def position(self) -> np.ndarray:
        if self.game_object and self.game_object.transform:
            return self.game_object.transform.position
        return self._fallback_position.copy()
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        if self.game_object and self.game_object.transform:
            self.game_object.transform.position = value
        else:
            self._fallback_position = np.array(value, dtype=np.float32)
    
    @property
    def x(self) -> float:
        return float(self.position[0])
    
    @x.setter
    def x(self, value: float):
        p = self.position
        p[0] = value
        self.position = p
    
    @property
    def y(self) -> float:
        return float(self.position[1])
    
    @y.setter
    def y(self, value: float):
        p = self.position
        p[1] = value
        self.position = p
    
    @property
    def z(self) -> float:
        return float(self.position[2])
    
    @z.setter
    def z(self, value: float):
        p = self.position
        p[2] = value
        self.position = p
