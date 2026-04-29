"""
Scene3D - A scene that can be shown in a Window3D.
Similar to arcade.View, but renamed for clarity.
"""
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Callable
import json
import pygame
import threading
import time

from engine3d.engine3d.gameobject import GameObject
from engine3d.engine3d.object3d import Object3D
from engine3d.engine3d.camera import Camera3D
from engine3d.engine3d.light import DirectionalLight3D, PointLight3D, Light3D
from engine3d.types import Color, ColorType
from engine3d.engine3d.ui.manager import UIManager

if TYPE_CHECKING:
    from .window import Window3D


class Scene3D:
    """
    A 3D scene that can be displayed in a Window3D.
    
    Subclass this to create different scenes (menu, game, pause screen, etc.)
    
    Example:
        class GameScene(Scene3D):
            def setup(self):
                self.player = self.add_object("player.obj")
                
            def on_update(self):
                self.player.rotation_y += Time.delta_time * 30
                
            def on_key_press(self, key, modifiers):
                if key == Keys.ESCAPE:
                    self.window.show_scene(MenuScene())
        
        class MenuScene(Scene3D):
            def setup(self):
                self.title = self.add_object("title.obj")
    """
    
    def __init__(self):
        """Initialize the scene."""
        self.window: Optional['Window3D'] = None
        self.objects: List[GameObject] = []
        
        # Camera setup
        self._cameras: List[Camera3D] = []
        self._main_camera: Optional[Camera3D] = None
        
        # Create default main camera
        cam_obj = GameObject("Main Camera")
        camera = Camera3D(is_main=True)  # Mark as main camera
        cam_obj.add_component(camera)
        cam_obj.transform.position = (0, 5, 10)
        cam_obj.transform.look_at((0, 0, 0))
        self.add_object(cam_obj)
        self._main_camera = camera
        
        self._setup_done = False
        self.canvas = UIManager(self)  # UI canvas for this scene
    
    @property
    def main_camera(self) -> Camera3D:
        """Get the main camera."""
        if self._main_camera:
            return self._main_camera
        # If no main camera, find first camera with is_main flag
        for cam in self._cameras:
            if cam.is_main:
                self._main_camera = cam
                return cam
        # If no main flag, find first camera
        if self._cameras:
            return self._cameras[0]
        # Fallback (shouldn't happen if initialized correctly)
        cam = Camera3D()
        return cam

    @main_camera.setter
    def main_camera(self, camera: Camera3D):
        """Set the main camera."""
        # Clear is_main flag on all existing cameras
        for cam in self._cameras:
            cam._is_main = False
        
        # Set the new main camera
        if camera in self._cameras:
            camera._is_main = True
            self._main_camera = camera
        else:
            # Auto-add if the camera's game object is in the scene
            if camera.game_object and camera.game_object in self.objects:
                self._cameras.append(camera)
                camera._is_main = True
                self._main_camera = camera

    @property
    def camera(self) -> Camera3D:
        """Alias for main_camera (backward compatibility)."""
        return self.main_camera
    
    @camera.setter
    def camera(self, value: Camera3D):
        """Set main camera (backward compatibility)."""
        self.main_camera = value
    
    @property
    def cameras(self) -> List[Camera3D]:
        """Get all cameras in the scene."""
        return self._cameras.copy()
    
    def get_cameras_sorted(self) -> List[Camera3D]:
        """
        Get all cameras sorted by priority (render order).
        
        Lower priority values render first (background).
        Higher priority values render last (overlay/on top).
        
        Returns:
            List of cameras sorted by priority (ascending).
        """
        return sorted(self._cameras, key=lambda cam: cam.priority)
    
    def add_camera(self, 
                   name: str = "Camera",
                   position: Tuple[float, float, float] = (0, 0, 0),
                   look_at: Optional[Tuple[float, float, float]] = None,
                   fov: float = 60.0,
                   viewport = None,
                   priority: int = 0,
                   is_main: bool = False) -> Camera3D:
        """
        Create and add a new camera to the scene.
        
        Args:
            name: Name for the camera GameObject
            position: Camera position in world space
            look_at: Point the camera should look at (optional)
            fov: Field of view in degrees
            viewport: Viewport for this camera (Viewport object or None for fullscreen)
            priority: Render priority (lower = render first)
            is_main: Whether this is the main camera
            
        Returns:
            The created Camera3D component
        """
        cam_obj = GameObject(name)
        camera = Camera3D(fov=fov, viewport=viewport, priority=priority, is_main=is_main)
        cam_obj.add_component(camera)
        cam_obj.transform.position = position
        
        if look_at:
            cam_obj.transform.look_at(look_at)
        
        self.add_object(cam_obj)
        
        if is_main:
            self.main_camera = camera
        
        return camera
    
    def create_minimap_camera(self,
                              position: Tuple[float, float, float] = (0, 50, 0),
                              look_at: Tuple[float, float, float] = (0, 0, 0),
                              corner: str = 'top-right',
                              size: float = 0.25,
                              fov: float = 60.0) -> Camera3D:
        """
        Create a minimap camera (top-down view in a corner).
        
        Args:
            position: Camera position (high up for top-down view)
            look_at: Point to look at (usually player or scene center)
            corner: 'top-right', 'top-left', 'bottom-right', or 'bottom-left'
            size: Viewport size (0.0 to 1.0)
            fov: Field of view
            
        Returns:
            The created Camera3D component
        """
        from engine3d.engine3d.camera import Viewport
        viewport = Viewport.minimap(corner, size)
        return self.add_camera(
            name="Minimap Camera",
            position=position,
            look_at=look_at,
            fov=fov,
            viewport=viewport,
            priority=100  # Render on top
        )
    
    def create_mirror_camera(self,
                             position: Tuple[float, float, float] = (0, 2, -5),
                             look_at: Tuple[float, float, float] = (0, 2, 5),
                             position_str: str = 'top',
                             width: float = 0.3,
                             height: float = 0.15) -> Camera3D:
        """
        Create a rear-view mirror camera.
        
        Args:
            position: Camera position (behind the player)
            look_at: Point to look at (behind the player)
            position_str: 'top', 'top-left', or 'top-right'
            width: Viewport width (0.0 to 1.0)
            height: Viewport height (0.0 to 1.0)
            
        Returns:
            The created Camera3D component
        """
        from engine3d.engine3d.camera import Viewport
        viewport = Viewport.mirror(position_str, width, height)
        return self.add_camera(
            name="Mirror Camera",
            position=position,
            look_at=look_at,
            fov=60.0,
            viewport=viewport,
            priority=50  # Render after main but before UI
        )
    
    def remove_camera(self, camera: Camera3D):
        """
        Remove a camera from the scene.
        
        Args:
            camera: The camera to remove
        """
        if camera in self._cameras:
            self._cameras.remove(camera)
            if self._main_camera == camera:
                # Find new main camera
                self._main_camera = None
                for cam in self._cameras:
                    if cam.is_main:
                        self._main_camera = cam
                        break
                if self._main_camera is None and self._cameras:
                    self._main_camera = self._cameras[0]
            # Remove the game object if it only contains the camera
            if camera.game_object:
                go = camera.game_object
                if len(go.components) == 1:  # Only the camera
                    self.remove_object(go)

    @property
    def light(self) -> Optional[DirectionalLight3D]:
        """Get the first DirectionalLight3D component in the scene, or None if none exists."""
        for obj in self.objects:
            l = obj.get_component(DirectionalLight3D)
            if l:
                return l
        return None
    
    def get_shadow_casting_lights(self) -> List[Light3D]:
        """
        Get all lights that cast shadows.
        
        Returns directional lights first, then point lights.
        Limited to MAX_SHADOW_LIGHTS (4) for performance.
        
        Returns:
            List of Light3D objects with cast_shadows=True
        """
        from engine3d.engine3d.graphics.shadow import MAX_SHADOW_LIGHTS
        
        lights = []
        
        # First, get directional lights (they're cheaper to render)
        for obj in self.objects:
            dl = obj.get_component(DirectionalLight3D)
            if dl and getattr(dl, 'cast_shadows', False):
                lights.append(dl)
                if len(lights) >= MAX_SHADOW_LIGHTS:
                    return lights
        
        # Then, get point lights
        for obj in self.objects:
            pl = obj.get_component(PointLight3D)
            if pl and getattr(pl, 'cast_shadows', False):
                lights.append(pl)
                if len(lights) >= MAX_SHADOW_LIGHTS:
                    return lights
        
        return lights
    
    def get_all_directional_lights(self) -> List[DirectionalLight3D]:
        """Get all directional lights in the scene."""
        lights = []
        for obj in self.objects:
            dl = obj.get_component(DirectionalLight3D)
            if dl:
                lights.append(dl)
        return lights
    
    def get_all_point_lights(self) -> List[PointLight3D]:
        """Get all point lights in the scene."""
        lights = []
        for obj in self.objects:
            pl = obj.get_component(PointLight3D)
            if pl:
                lights.append(pl)
        return lights
    
    def _attach_window(self, window: 'Window3D'):
        """Called when scene is attached to a window."""
        self.window = window
        if not self._setup_done:
            self.setup()
            self._setup_done = True
    
    def _detach_window(self):
        """Called when scene is detached from window."""
        self.on_hide()
    
    # =========================================================================
    # Object management
    # =========================================================================
    
    def add_object(self, obj_or_filename, **kwargs) -> GameObject:
        position = kwargs.pop('position', None)
        rotation = kwargs.pop('rotation', None)
        scale = kwargs.pop('scale', None)

        if isinstance(obj_or_filename, GameObject):
            go = obj_or_filename
        elif isinstance(obj_or_filename, Object3D):
            go = GameObject()
            go.add_component(obj_or_filename)
        else:
            go = GameObject()
            obj3d = Object3D(obj_or_filename, **kwargs)
            go.add_component(obj3d)
        
        if position is not None:
            go.transform.position = position
        if rotation is not None:
            go.transform.rotation = rotation
        if scale is not None:
            go.transform.scale = scale
            
        self.objects.append(go)
        
        # Set scene reference on the GameObject for components like ParticleSystem
        go._scene = self
        
        # Initialize GPU if window is available
        if self.window and self.window._ctx:
            obj3d_comp = go.get_component(Object3D)
            if obj3d_comp:
                self.window._ensure_mesh(obj3d_comp)
        
        # Note: awake_scripts() and start_scripts() should NOT be called here
        # They should only be called when play mode begins
        
        # Register cameras
        for cam in go.get_components(Camera3D):
            if cam not in self._cameras:
                self._cameras.append(cam)
                if self._main_camera is None:
                    self._main_camera = cam
        
        return go
    
    def remove_object(self, obj: GameObject):
        """Remove object from scene, including all its children recursively."""
        if obj not in self.objects:
            return
        
        # Collect all descendants (children, grandchildren, etc.) first
        descendants = []
        def collect_descendants(transform):
            for child in transform.children:  # children is a property, not a method
                if child.game_object in self.objects:
                    descendants.append(child.game_object)
                    collect_descendants(child)
        
        collect_descendants(obj.transform)
        
        # Remove all descendants first (bottom-up)
        for descendant in descendants:
            if descendant in self.objects:
                # Release GPU resources
                desc_obj3d = descendant.get_component(Object3D)
                if desc_obj3d:
                    if self.window:
                        self.window._release_mesh(desc_obj3d)
                    else:
                        desc_obj3d._release_gpu()
                
                # Unregister cameras
                for cam in descendant.get_components(Camera3D):
                    if cam in self._cameras:
                        self._cameras.remove(cam)
                        if self._main_camera == cam:
                            self._main_camera = self._cameras[0] if self._cameras else None
                
                self.objects.remove(descendant)
        
        # Now remove the main object
        obj3d = obj.get_component(Object3D)
        if obj3d:
            if self.window:
                self.window._release_mesh(obj3d)
            else:
                obj3d._release_gpu()
        
        # Unregister cameras
        for cam in obj.get_components(Camera3D):
            if cam in self._cameras:
                self._cameras.remove(cam)
                if self._main_camera == cam:
                    self._main_camera = self._cameras[0] if self._cameras else None

        self.objects.remove(obj)
        
        # Clear scene reference
        if hasattr(obj, '_scene'):
            obj._scene = None
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            obj3d = obj.get_component(Object3D)
            if obj3d:
                if self.window:
                    self.window._release_mesh(obj3d)
                else:
                    obj3d._release_gpu()
            if hasattr(obj, '_scene'):
                obj._scene = None
        self.objects.clear()
        self._cameras.clear()
        self._main_camera = None
    
    def get_objects_by_name(self, name: str) -> List[GameObject]:
        """Get all objects with a specific name."""
        return [obj for obj in self.objects if obj.name == name]
    
    def get_objects_by_tag(self, tag: str) -> List[GameObject]:
        """Get all objects with a specific tag."""
        return [obj for obj in self.objects if obj.tag == tag]
    
    def load_object(self, filename: str, **kwargs) -> GameObject:
        """
        Load and add a 3D object from file.
        
        Alias for add_object() with a filename.
        """
        return self.add_object(filename, **kwargs)

    # =========================================================================
    # Lifecycle methods (override these in subclass)
    # =========================================================================
    
    def setup(self):
        """
        Called once when the scene is first shown.
        Override to set up your scene.
        """
        # Add default directional light if none exists
        if self.light is None:
            light_obj = GameObject("Directional Light")
            light_obj.add_component(DirectionalLight3D())
            light_obj.transform.rotation = (-45, 30, 0)
            self.add_object(light_obj)
    
    def on_show(self):
        """
        Called each time the scene becomes active.
        """
        pass
    
    def on_hide(self):
        """
        Called when switching to a different scene.
        """
        pass
    
    def on_update(self):
        """
        Called every frame to update the scene.
        """
        pass
    
    def on_draw(self):
        """
        Called after the scene is rendered.
        Override to add custom drawing (UI, etc.)
        """
        pass
    
    # =========================================================================
    # Input methods (override these in subclass)
    # =========================================================================
    
    def on_key_press(self, key: int, modifiers: int):
        """
        Called when a key is pressed.
        
        Args:
            key: Key code (use Keys constants)
            modifiers: Modifier flags (shift, ctrl, alt)
        """
        pass
    
    def on_key_release(self, key: int, modifiers: int):
        """
        Called when a key is released.
        """
        pass
    
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        """
        Called when a mouse button is pressed.
        
        Args:
            x, y: Mouse position
            button: Button number (1=left, 2=middle, 3=right)
            modifiers: Modifier flags
        """
        pass
    
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        """
        Called when a mouse button is released.
        """
        pass
    
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        """
        Called when the mouse moves.
        
        Args:
            x, y: Current mouse position
            dx, dy: Change in position since last call
        """
        pass
    
    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        """
        Called when the mouse wheel is scrolled.
        
        Args:
            x, y: Mouse position
            scroll_x, scroll_y: Scroll amounts
        """
        pass

    def on_resize(self, width: int, height: int):
        """Called when the window is resized."""
        pass

    # =========================================================================
    # Serialization
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Save this scene (camera, light, objects) to a scene file.
        """
        data = self._to_scene_dict()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scene3D":
        """
        Load a scene file and return the created Scene3D.
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls._from_scene_dict(data)

    def _to_scene_dict(self) -> dict:
        # Handle Vector3 or numpy array for camera position/target
        cam_pos = self.camera.position
        cam_target = self.camera.target
        if hasattr(cam_pos, 'to_list'):
            cam_pos = cam_pos.to_list()
        elif hasattr(cam_pos, 'tolist'):
            cam_pos = cam_pos.tolist()
        if hasattr(cam_target, 'to_list'):
            cam_target = cam_target.to_list()
        elif hasattr(cam_target, 'tolist'):
            cam_target = cam_target.tolist()
            
        # Filter out internal particle system particles (they're managed by ParticleSystem)
        visible_objects = [obj for obj in self.objects 
                           if not getattr(obj, '_is_particle_system_particle', False)]
        
        return {
            "camera": {
                "position": cam_pos,
                "target": cam_target,
                "fov": self.camera.fov,
                "near": self.camera.near,
                "far": self.camera.far,
            },
            "objects": [obj._to_prefab_dict() for obj in visible_objects],
        }

    def clone(self) -> "Scene3D":
        """
        Create a deep copy of this scene.
        """
        data = self._to_scene_dict()
        new_scene = self.__class__._from_scene_dict(data)
        # Copy editor specific labels if any
        if hasattr(self, 'editor_label'):
            new_scene.editor_label = self.editor_label
        return new_scene

    @classmethod
    def _from_scene_dict(cls, data: dict) -> "Scene3D":
        scene = cls()
        scene.clear_objects()
        camera_data = data.get("camera", {})
        if camera_data:
            scene.camera.position = camera_data.get("position", scene.camera.position)
            scene.camera.target = camera_data.get("target", scene.camera.target)
            scene.camera.fov = camera_data.get("fov", scene.camera.fov)
            scene.camera.near = camera_data.get("near", scene.camera.near)
            scene.camera.far = camera_data.get("far", scene.camera.far)

        # First pass: Create all game objects and build registry
        go_registry: Dict[str, GameObject] = {}
        for obj_data in data.get("objects", []):
            obj = GameObject._from_prefab_dict(obj_data)
            obj._scene = scene  # Set scene reference so components can find it
            scene.objects.append(obj)
            go_registry[obj._id] = obj

        # Second pass: Resolve component references in all objects
        for obj in scene.objects:
            cls._resolve_component_references(obj, go_registry)

        # Third pass: Fix ParticleSystem containers that were attached to the
        # wrong scene during on_attach() (on_attach runs inside _from_prefab_dict
        # before _scene is set, so the container may point to the old/current scene)
        from .particle import ParticleSystem
        for obj in scene.objects:
            for comp in obj.components:
                if isinstance(comp, ParticleSystem) and comp._container is not scene:
                    # Remove any particles that were built in the wrong container
                    old_container = comp._container
                    for p in comp._particles:
                        if (old_container is not None
                                and hasattr(old_container, 'objects')
                                and p.obj in old_container.objects):
                            old_container.objects.remove(p.obj)
                    comp._particles = []
                    comp._container = scene

        for obj in scene.objects:
            for cam in obj.get_components(Camera3D):
                if cam not in scene._cameras:
                    scene._cameras.append(cam)
                    if scene._main_camera is None:
                        scene._main_camera = cam

        light_data = data.get("light", {})
        if light_data:
            # Handle legacy light data by creating a new GameObject
            light_obj = GameObject("Legacy Light")
            light = DirectionalLight3D()
            light.direction = light_data.get("direction", light.direction)
            light.color = light_data.get("color", light.color)
            light.intensity = light_data.get("intensity", light.intensity)
            light.ambient = light_data.get("ambient", light.ambient)
            light_obj.add_component(light)
            scene.objects.append(light_obj)
            
            # Register the light's camera if it has one
            for cam in light_obj.get_components(Camera3D):
                if cam not in scene._cameras:
                    scene._cameras.append(cam)
                    if scene._main_camera is None:
                        scene._main_camera = cam

        # Note: awake_scripts() and start_scripts() should NOT be called here
        # They should only be called when play mode begins

        return scene
    
    @staticmethod
    def _resolve_component_references(obj: GameObject, go_registry: Dict[str, GameObject]) -> None:
        """
        Second pass: Resolve component references in all components of an object.
        This is needed because component references point to other objects/components
        that may not exist during the first pass of deserialization.
        """
        from .gameobject import GameObject
        
        for component in obj.components:
            # Check if this component has serialized state that needs resolution
            serialized_state = getattr(component, '_serialized_state', None)
            if serialized_state:
                # Re-deserialize with the registry to resolve component references
                restored_state = GameObject._deserialize_value(serialized_state, go_registry)
                component.__dict__.update(restored_state)
                # Clean up the temporary attribute
                delattr(component, '_serialized_state')

    # 2D drawing methods (forward to window if attached)
    # Allows drawing shapes and text in Scene3D.on_draw()
    def draw_text(self, text: str, x: int, y: int, color: ColorType = Color.WHITE,
                  font_size: int = 24, font_name: Optional[str] = None,
                  anchor_x: str = 'left', anchor_y: str = 'top',
                  baseline_adjust: bool = True) -> None:
        """Draw text (y=top of bounding box; delegates to window)."""
        if self.window:
            self.window.draw_text(text, x, y, color, font_size, font_name, anchor_x, anchor_y, baseline_adjust)
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int,
                       color: ColorType, border_width: int = 0) -> None:
        """Draw rectangle (delegates to window)."""
        if self.window:
            self.window.draw_rectangle(x, y, width, height, color, border_width)
    
    def draw_circle(self, x: int, y: int, radius: int, color: ColorType,
                    border_width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw circle (delegates to window)."""
        if self.window:
            self.window.draw_circle(x, y, radius, color, border_width, aa)
    
    def draw_ellipse(self, x: int, y: int, width: int, height: int,
                     color: ColorType, border_width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw ellipse (delegates to window)."""
        if self.window:
            self.window.draw_ellipse(x, y, width, height, color, border_width, aa)
    
    def draw_polygon(self, points: List[Tuple[int, int]], color: ColorType,
                     border_width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw polygon (delegates to window)."""
        if self.window:
            self.window.draw_polygon(points, color, border_width, aa)
    
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int],
                  color: ColorType, width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw line (delegates to window)."""
        if self.window:
            self.window.draw_line(start, end, color, width, aa)
    
    def draw_image(self, image: Union[str, 'pygame.Surface'], x: int, y: int,
                   scale: float = 1.0, alpha: float = 1.0) -> None:
        """Draw image (path or Surface; delegates to window)."""
        if self.window:
            self.window.draw_image(image, x, y, scale, alpha)


class SceneManager:
    """
    Manages scene loading with both synchronous and asynchronous options.
    
    Similar to Unity's SceneManager, providing progress callbacks for async loading.
    
    Create an instance to manage loading state per-manager rather than globally:
        manager = SceneManager()
        manager.load_scene_async("level1.scene", on_complete=window.show_scene)
    """
    
    # Callback type: progress (0.0 to 1.0)
    ProgressCallback = Callable[[float], None]
    
    def __init__(self):
        self._current_load: Optional[threading.Thread] = None
        self._loading_progress: float = 0.0
        self._loaded_scene: Optional[Scene3D] = None
        self._loading_error: Optional[Exception] = None
        self._pending_callbacks: list = []
        self._lock = threading.Lock()
    
    def poll(self) -> None:
        """Call from the main thread (e.g. in on_update) to dispatch queued callbacks."""
        with self._lock:
            callbacks = list(self._pending_callbacks)
            self._pending_callbacks.clear()
        for cb in callbacks:
            cb()
    
    @staticmethod
    def load_scene(path: str) -> Scene3D:
        """
        Synchronously load a scene from file.
        
        This will block until the scene is fully loaded. For large scenes,
        this may cause frame drops. Use load_scene_async for non-blocking loads.
        
        Args:
            path: Path to the .scene file
            
        Returns:
            The loaded Scene3D instance
            
        Example:
            scene = SceneManager.load_scene("Scenes/level1.scene")
            window.show_scene(scene)
        """
        return Scene3D.load(path)
    
    def load_scene_async(self, path: str, 
                         on_progress: Optional[ProgressCallback] = None,
                         on_complete: Optional[Callable[[Scene3D], None]] = None,
                         on_error: Optional[Callable[[Exception], None]] = None) -> None:
        """
        Asynchronously load a scene from file with progress callbacks.
        
        The scene loads in a background thread. on_complete and on_error are
        queued and dispatched on the main thread when you call poll().
        on_progress is called from the background thread (safe for simple
        assignments like updating a progress float).
        
        Args:
            path: Path to the .scene file
            on_progress: Callback(progress: float) called during loading (0.0 to 1.0)
            on_complete: Callback(scene: Scene3D) called when loading completes (via poll)
            on_error: Callback(error: Exception) called if loading fails (via poll)
            
        Example:
            manager = SceneManager()
            
            def on_complete(scene):
                window.show_scene(scene)
            
            manager.load_scene_async("Scenes/level1.scene", on_complete=on_complete)
            # In your update loop: manager.poll()
        """
        # Reset state
        self._loading_progress = 0.0
        self._loaded_scene = None
        self._loading_error = None
        
        def load_in_background():
            try:
                # Step 1: Read file (10%)
                self._loading_progress = 0.1
                if on_progress:
                    on_progress(self._loading_progress)
                
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                
                # Step 2: Create scene object (30%)
                self._loading_progress = 0.3
                if on_progress:
                    on_progress(self._loading_progress)
                time.sleep(0.01)  # Small delay to allow UI update
                
                scene = Scene3D.__new__(Scene3D)
                scene.window = None
                scene.objects = []
                scene._cameras = []
                scene._main_camera = None
                scene._setup_done = False
                scene.canvas = UIManager(scene)
                
                # Step 3: Setup camera (50%)
                self._loading_progress = 0.5
                if on_progress:
                    on_progress(self._loading_progress)
                time.sleep(0.01)
                
                camera_data = data.get("camera", {})
                if camera_data:
                    cam_obj = GameObject("Main Camera")
                    camera = Camera3D()
                    cam_obj.add_component(camera)
                    cam_obj.transform.position = camera_data.get("position", (0, 5, 10))
                    cam_obj.transform.look_at(camera_data.get("target", (0, 0, 0)))
                    camera.fov = camera_data.get("fov", 60)
                    camera.near = camera_data.get("near", 0.1)
                    camera.far = camera_data.get("far", 1000)
                    scene.add_object(cam_obj)
                    scene._main_camera = camera
                
                # Step 4: Load objects (70%)
                self._loading_progress = 0.7
                if on_progress:
                    on_progress(self._loading_progress)
                time.sleep(0.01)
                
                objects_data = data.get("objects", [])
                total_objects = len(objects_data)
                
                for i, obj_data in enumerate(objects_data):
                    obj = GameObject._from_prefab_dict(obj_data)
                    scene.objects.append(obj)
                    
                    # Update progress within object loading (70% to 90%)
                    obj_progress = 0.7 + (0.2 * (i + 1) / total_objects) if total_objects > 0 else 0.9
                    self._loading_progress = obj_progress
                    if on_progress:
                        on_progress(self._loading_progress)
                
                # Step 5: Resolve references and finalize (100%)
                self._loading_progress = 0.95
                if on_progress:
                    on_progress(self._loading_progress)
                
                # Build registry and resolve references
                go_registry = {obj._id: obj for obj in scene.objects}
                for obj in scene.objects:
                    Scene3D._resolve_component_references(obj, go_registry)
                
                # Register cameras
                for obj in scene.objects:
                    for cam in obj.get_components(Camera3D):
                        if cam not in scene._cameras:
                            scene._cameras.append(cam)
                            if scene._main_camera is None:
                                scene._main_camera = cam
                
                self._loading_progress = 1.0
                self._loaded_scene = scene
                
                if on_progress:
                    on_progress(1.0)
                
                # Queue on_complete for main thread dispatch via poll()
                if on_complete:
                    with self._lock:
                        self._pending_callbacks.append(lambda: on_complete(scene))
                    
            except Exception as e:
                self._loading_error = e
                # Queue on_error for main thread dispatch via poll()
                if on_error:
                    with self._lock:
                        self._pending_callbacks.append(lambda: on_error(e))
        
        # Start loading in background thread
        self._current_load = threading.Thread(target=load_in_background, daemon=True)
        self._current_load.start()
    
    def get_loading_progress(self) -> float:
        """
        Get the current loading progress (0.0 to 1.0).
        
        Returns 1.0 if no loading is in progress.
        """
        if self._current_load is None or not self._current_load.is_alive():
            return 1.0 if self._loaded_scene is not None else 0.0
        return self._loading_progress
    
    def is_loading(self) -> bool:
        """Check if a scene is currently being loaded."""
        return self._current_load is not None and self._current_load.is_alive()
    
    def get_loaded_scene(self) -> Optional[Scene3D]:
        """
        Get the scene that was loaded by the most recent async operation.
        Returns None if no scene has been loaded.
        """
        return self._loaded_scene
    
    def get_loading_error(self) -> Optional[Exception]:
        """Get any error that occurred during the last async load operation."""
        return self._loading_error
