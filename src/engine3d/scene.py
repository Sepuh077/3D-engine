"""
Scene3D - A scene that can be shown in a Window3D.
Similar to arcade.View, but renamed for clarity.
"""
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import json, pygame

from .gameobject import GameObject
from .object3d import Object3D
from .camera import Camera3D
from .light import DirectionalLight3D
from .graphics.color import Color, ColorType
from .ui.manager import UIManager

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
        camera = Camera3D()
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
        # If no main camera, find first camera
        if self._cameras:
            return self._cameras[0]
        # Fallback (shouldn't happen if initialized correctly)
        cam = Camera3D()
        return cam

    @main_camera.setter
    def main_camera(self, camera: Camera3D):
        """Set the main camera."""
        if camera in self._cameras:
            self._main_camera = camera
        else:
            # Maybe auto-add?
            if camera.game_object and camera.game_object in self.objects:
                self._cameras.append(camera)
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
    def light(self) -> Optional[DirectionalLight3D]:
        """Get the first DirectionalLight3D component in the scene, or None if none exists."""
        for obj in self.objects:
            l = obj.get_component(DirectionalLight3D)
            if l:
                return l
        return None
    
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
        
        # Initialize GPU if window is available
        if self.window and self.window._ctx:
            obj3d_comp = go.get_component(Object3D)
            if obj3d_comp:
                self.window._ensure_mesh(obj3d_comp)
        
        # Start scripts on the new object
        go.start_scripts()
        
        # Register cameras
        for cam in go.get_components(Camera3D):
            if cam not in self._cameras:
                self._cameras.append(cam)
                if self._main_camera is None:
                    self._main_camera = cam
        
        return go
    
    def remove_object(self, obj: GameObject):
        """Remove object from scene."""
        if obj in self.objects:
            if self.window:
                if obj.get_component(Object3D):
                    self.window._release_mesh(obj.get_component(Object3D))
            else:
                if obj.get_component(Object3D):
                    obj.get_component(Object3D)._release_gpu()
            
            # Unregister cameras
            for cam in obj.get_components(Camera3D):
                if cam in self._cameras:
                    self._cameras.remove(cam)
                    if self._main_camera == cam:
                        self._main_camera = self._cameras[0] if self._cameras else None

            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            if self.window:
                if obj.get_component(Object3D):
                    self.window._release_mesh(obj.get_component(Object3D))
            else:
                if obj.get_component(Object3D):
                    obj.get_component(Object3D)._release_gpu()
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
        # Add default directional light
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
        # Don't serialize system objects (Main Camera, Directional Light)
        # to avoid duplicates when restoring
        system_objects = {"Main Camera", "Directional Light"}
        
        # Save directional light settings
        light_data = {}
        if self.light:
            light_data = {
                "direction": self.light.direction.tolist() if hasattr(self.light.direction, 'tolist') else list(self.light.direction),
                "color": self.light.color.tolist() if hasattr(self.light.color, 'tolist') else list(self.light.color),
                "intensity": self.light.intensity,
                "ambient": self.light.ambient,
            }
        
        return {
            "camera": {
                "position": self.camera.position.tolist(),
                "target": self.camera.target.tolist(),
                "fov": self.camera.fov,
                "near": self.camera.near,
                "far": self.camera.far,
            },
            "light": light_data,
            "objects": [obj._to_prefab_dict() for obj in self.objects if obj.name not in system_objects],
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
        camera_data = data.get("camera", {})
        if camera_data:
            # Update the existing Main Camera with saved settings
            scene.camera.position = camera_data.get("position", scene.camera.position)
            scene.camera.target = camera_data.get("target", scene.camera.target)
            scene.camera.fov = camera_data.get("fov", scene.camera.fov)
            scene.camera.near = camera_data.get("near", scene.camera.near)
            scene.camera.far = camera_data.get("far", scene.camera.far)

        # Update the existing Directional Light with saved settings if available
        light_data = data.get("light", {})
        if light_data:
            # Find the existing Directional Light and update it
            for obj in scene.objects:
                light = obj.get_component(DirectionalLight3D)
                if light:
                    light.direction = light_data.get("direction", light.direction)
                    light.color = light_data.get("color", light.color)
                    light.intensity = light_data.get("intensity", light.intensity)
                    light.ambient = light_data.get("ambient", light.ambient)
                    break

        # Add user-created objects
        for obj_data in data.get("objects", []):
            scene.objects.append(GameObject._from_prefab_dict(obj_data))

        for obj in scene.objects:
            obj.start_scripts()

        return scene

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
