"""
Shadow mapping system for directional lights.

Implements basic shadow mapping with orthographic projection for directional lights.
"""
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

import moderngl

if TYPE_CHECKING:
    from .light import DirectionalLight3D
    from .camera import Camera3D


class ShadowMap:
    """
    Manages shadow map texture and framebuffer for a directional light.
    
    Shadow mapping works by rendering the scene from the light's perspective
    into a depth texture. During the main pass, we compare each fragment's
    depth (from the light's view) to determine if it's in shadow.
    """
    
    def __init__(self, ctx: moderngl.Context, resolution: int = 1024):
        """
        Initialize shadow map.
        
        Args:
            ctx: ModernGL context
            resolution: Shadow map resolution (width and height)
        """
        self.ctx = ctx
        self.resolution = resolution
        
        # Create depth-only texture for shadow map
        self.depth_texture = ctx.depth_texture((resolution, resolution))
        
        # Configure depth texture for shadow sampling
        # Set nearest filtering to avoid shadow acne from interpolation
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # Clamp to edge to avoid artifacts at borders
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False
        
        # Create framebuffer with depth attachment only
        self.framebuffer = ctx.framebuffer(depth_attachment=self.depth_texture)
        
        # Cached light space matrix
        self._light_space_matrix: Optional[np.ndarray] = None
        
        # Store original viewport for restoration
        self._original_viewport: Optional[Tuple[int, int, int, int]] = None
    
    def begin_pass(self):
        """
        Begin shadow rendering pass.
        
        Binds the shadow framebuffer and sets up viewport.
        Call this before rendering objects from light's view.
        """
        self._original_viewport = self.ctx.viewport
        self.framebuffer.use()
        self.ctx.viewport = (0, 0, self.resolution, self.resolution)
        
        # Clear depth buffer to 1.0 (far plane)
        self.ctx.clear(depth=1.0)
        
        # Enable depth testing for shadow pass
        self.ctx.enable(self.ctx.DEPTH_TEST)
    
    def end_pass(self):
        """
        End shadow rendering pass.
        
        Restores the original framebuffer and viewport.
        """
        # Restore original viewport
        if self._original_viewport is not None:
            self.ctx.viewport = self._original_viewport
        
        # Note: The default framebuffer will be bound by the next render call
        # We don't explicitly unbind here to avoid issues with different contexts
    
    def calculate_light_space_matrix(
        self,
        light: 'DirectionalLight3D',
        camera: 'Camera3D',
        shadow_distance: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate the light's view-projection matrix for shadow mapping.
        
        This creates an orthographic projection that encompasses the visible
        scene from the camera's perspective, as seen from the light.
        
        Args:
            light: The directional light
            camera: The main camera (used to determine shadow coverage)
            shadow_distance: Maximum distance from camera for shadows
            
        Returns:
            4x4 light space matrix (projection * view)
        """
        if shadow_distance is None:
            shadow_distance = light.shadow_distance
        
        # Get light direction (where light is going)
        light_dir = np.array(light.direction.to_tuple(), dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Get camera position for shadow centering
        cam_pos = np.array(camera.position.to_tuple(), dtype=np.float32)
        
        # Shadow region center - place it between camera and a point in front
        shadow_center = cam_pos
        
        # For the view matrix, we need to look FROM the light position IN the direction of the light
        # The light position should be behind the shadow region (opposite to light direction)
        light_pos = shadow_center - light_dir * shadow_distance
        
        # The look direction is the same as the light direction (where the light is pointing)
        look_dir = light_dir
        
        # Calculate orthonormal basis for light's coordinate system
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Handle case where light direction is parallel to world up
        if abs(np.dot(look_dir, world_up)) > 0.999:
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Right vector: perpendicular to look_dir and world_up
        # Note: cross(world_up, look_dir) gives right vector for a right-handed system
        light_right = np.cross(world_up, look_dir)
        light_right = light_right / np.linalg.norm(light_right)
        
        # Up vector: perpendicular to look_dir and right
        # Recalculate up to ensure orthogonality
        light_up = np.cross(look_dir, light_right)
        light_up = light_up / np.linalg.norm(light_up)
        
        # Build view matrix (column-major for OpenGL)
        # The view matrix transforms world coordinates to light's view space
        # In OpenGL, camera looks down -Z axis in view space
        # So the third row of the rotation part is -look_dir
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = light_right
        view[1, :3] = light_up
        view[2, :3] = -look_dir  # OpenGL convention: camera looks down -Z
        # Translation: -dot(right, pos), -dot(up, pos), dot(forward, pos)
        view[0, 3] = -np.dot(light_right, light_pos)
        view[1, 3] = -np.dot(light_up, light_pos)
        view[2, 3] = np.dot(look_dir, light_pos)
        
        # Build orthographic projection matrix
        # Size the projection to cover the shadow region
        half_size = shadow_distance
        
        # Depth range - needs to cover from light position to beyond shadow region
        near = 0.1
        far = shadow_distance * 2.0
        
        left = -half_size
        right = half_size
        bottom = -half_size
        top = half_size
        
        # Standard OpenGL orthographic projection matrix
        # Maps [left, right] x [bottom, top] x [near, far] to [-1, 1] x [-1, 1] x [-1, 1]
        # Note: OpenGL uses right-handed coordinate system, so near maps to -1, far maps to 1
        projection = np.eye(4, dtype=np.float32)
        projection[0, 0] = 2.0 / (right - left)
        projection[1, 1] = 2.0 / (top - bottom)
        projection[2, 2] = -2.0 / (far - near)
        projection[0, 3] = -(right + left) / (right - left)
        projection[1, 3] = -(top + bottom) / (top - bottom)
        projection[2, 3] = -(far + near) / (far - near)
        
        self._light_space_matrix = projection @ view
        return self._light_space_matrix
    
    def get_light_space_matrix(self) -> Optional[np.ndarray]:
        """Get the cached light space matrix."""
        return self._light_space_matrix
    
    def release(self):
        """Release GPU resources."""
        if self.depth_texture is not None:
            self.depth_texture.release()
            self.depth_texture = None
        if self.framebuffer is not None:
            self.framebuffer.release()
            self.framebuffer = None


def calculate_shadow_bounds(
    camera: 'Camera3D',
    shadow_distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the bounding box of the shadow region in world space.
    
    Args:
        camera: The camera
        shadow_distance: Maximum shadow distance
        
    Returns:
        Tuple of (min_corner, max_corner) in world space
    """
    cam_pos = np.array(camera.position.to_tuple(), dtype=np.float32)
    cam_forward = np.array(camera.forward, dtype=np.float32)
    cam_right = np.array(camera.right, dtype=np.float32)
    cam_up = np.array(camera.up, dtype=np.float32)
    
    # Calculate frustum corners at shadow_distance
    half_dist = shadow_distance * 0.5
    
    # Create 8 corners of the shadow region
    corners = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                corner = cam_pos + cam_forward * half_dist
                corner += cam_right * (x * half_dist)
                corner += cam_up * (y * half_dist)
                corner += cam_forward * (z * half_dist * 0.5)
                corners.append(corner)
    
    corners = np.array(corners)
    return corners.min(axis=0), corners.max(axis=0)
