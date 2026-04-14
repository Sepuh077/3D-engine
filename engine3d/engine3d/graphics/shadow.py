"""
Shadow mapping support for the 3D engine.

Implements basic shadow mapping for directional lights using depth-only rendering
from the light's perspective.
"""
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl


class ShadowMap:
    """
    Manages a shadow map framebuffer for shadow rendering.
    
    Creates a depth texture that stores the depth values from the light's perspective.
    This is then sampled during the main pass to determine if fragments are in shadow.
    """
    
    def __init__(self, ctx: 'moderngl.Context', resolution: int = 1024):
        """
        Initialize the shadow map.
        
        Args:
            ctx: ModernGL context
            resolution: Shadow map resolution (width and height)
        """
        self.ctx = ctx
        self.resolution = resolution
        
        # Create depth-only texture for shadow map
        self.depth_texture = ctx.depth_texture((resolution, resolution))
        self.depth_texture.compare_func = '<='  # Depth comparison for shadow sampling
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False
        
        # Create framebuffer with depth attachment only
        self.framebuffer = ctx.framebuffer(depth_attachment=self.depth_texture)
        
        # Store previous viewport/FBO for restoration (supports editor custom FBO)
        self._prev_viewport = None
        self._prev_fbo = None
    
    def begin(self):
        """
        Begin shadow pass - bind framebuffer and set viewport.
        
        Call this before rendering objects to the shadow map.
        """
        self._prev_viewport = self.ctx.viewport
        self._prev_fbo = self.ctx.detect_framebuffer()
        self.framebuffer.use()
        self.ctx.viewport = (0, 0, self.resolution, self.resolution)
        
        # Clear depth buffer
        self.ctx.clear(depth=1.0)
        
        # depth writes enabled by default for depth fb
    
    def end(self):
        """
        End shadow pass - restore previous framebuffer (editor custom FBO) and viewport.
        
        Call this after rendering all shadow casters.
        """
        # Restore previous FBO (critical for editor embedding, not just screen)
        if self._prev_fbo:
            self._prev_fbo.use()
        else:
            self.ctx.screen.use()
        
        # Restore viewport
        if self._prev_viewport:
            self.ctx.viewport = self._prev_viewport
    
    def use(self, location: int = 1):
        """
        Bind the shadow map texture for sampling in shaders.
        
        Args:
            location: Texture unit to bind to (default 1, since 0 is often used for other textures)
        """
        self.depth_texture.use(location=location)
    
    def release(self):
        """Release GPU resources."""
        if self.depth_texture:
            self.depth_texture.release()
            self.depth_texture = None
        if self.framebuffer:
            self.framebuffer.release()
            self.framebuffer = None




def calculate_light_space_matrix(
    light_direction: np.ndarray,
    shadow_distance: float = 50.0,
    scene_center: np.ndarray = None,
    scene_radius: float = 20.0
) -> np.ndarray:
    """
    Calculate the light space matrix (projection * view) for shadow rendering.
    
    For directional lights, this uses an orthographic projection that encompasses
    the visible scene area. Uses centered ortho for stability.
    
    Args:
        light_direction: Normalized direction vector pointing FROM the light
        shadow_distance: Distance from camera for shadow rendering
        scene_center: Center of the scene to shadow (default: origin)
        scene_radius: Approximate radius of the scene to encompass
    
    Returns:
        4x4 light space matrix (projection @ view)
    """
    if scene_center is None:
        scene_center = np.array([0.0, 0.0, 0.0])
    
    # Normalize light direction
    light_dir = np.array(light_direction, dtype=np.float32)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)
    
    # Position the light above the scene center
    light_pos = scene_center - light_dir * shadow_distance
    
    # Calculate view matrix (look at scene center from light position)
    # Up vector: try to use world up, but use another axis if light points straight up/down
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(light_dir, world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    
    # Calculate view matrix components
    forward = -light_dir  # Camera looks opposite to light direction
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(right, forward)
    
    # Build view matrix
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = forward
    view[0, 3] = -np.dot(right, light_pos)
    view[1, 3] = -np.dot(up, light_pos)
    view[2, 3] = -np.dot(forward, light_pos)
    
    # Orthographic projection for directional light - centered and stable
    ortho_size = max(scene_radius, shadow_distance * 0.6)
    
    left = -ortho_size
    right = ortho_size
    bottom = -ortho_size
    top = ortho_size
    near = 0.1
    far = shadow_distance * 2.0
    
    proj = np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    return proj @ view
