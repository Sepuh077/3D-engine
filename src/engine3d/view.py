"""
View3D - A scene/view that can be shown in a Window3D.
Similar to arcade.View.
"""
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from .gameobject import GameObject
from .object3d import Object3D
from .camera import Camera3D
from .light import Light3D
from .graphics.color import Color, ColorType
from .component import Time

if TYPE_CHECKING:
    from .window import Window3D


class View3D:
    """
    A 3D scene/view that can be displayed in a Window3D.
    
    Subclass this to create different scenes (menu, game, pause screen, etc.)
    
    Example:
        class GameView(View3D):
            def setup(self):
                self.player = self.add_object("player.obj")
                
            def on_update(self):
                self.player.rotation_y += Time.delta_time * 30
                
            def on_key_press(self, key, modifiers):
                if key == Keys.ESCAPE:
                    self.window.show_view(MenuView())
        
        class MenuView(View3D):
            def setup(self):
                self.title = self.add_object("title.obj")
    """
    
    def __init__(self):
        """Initialize the view."""
        self.window: Optional['Window3D'] = None
        self.objects: List[GameObject] = []
        self.camera = Camera3D()
        self._fallback_light = Light3D()
        self._setup_done = False
        
    @property
    def light(self) -> Light3D:
        """Get the first Light3D component in the view, or a fallback."""
        for obj in self.objects:
            l = obj.get_component(Light3D)
            if l:
                return l
        return self._fallback_light
    
    def _attach_window(self, window: 'Window3D'):
        """Called when view is attached to a window."""
        self.window = window
        if not self._setup_done:
            self.setup()
            self._setup_done = True
    
    def _detach_window(self):
        """Called when view is detached from window."""
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
        
        return go
    
    def remove_object(self, obj: GameObject):
        """Remove object from scene."""
        if obj in self.objects:
            if self.window:
                if obj.get_component(Object3D): self.window._release_mesh(obj.get_component(Object3D))
            else:
                if obj.get_component(Object3D): obj.get_component(Object3D)._release_gpu()
            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            if self.window:
                if obj.get_component(Object3D): self.window._release_mesh(obj.get_component(Object3D))
            else:
                if obj.get_component(Object3D): obj.get_component(Object3D)._release_gpu()
        self.objects.clear()
    
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
        Called once when the view is first shown.
        Override to set up your scene.
        """
        # Add default directional light
        light_obj = GameObject("Directional Light")
        light_obj.add_component(Light3D())
        light_obj.transform.rotation = (-45, 30, 0)
        self.add_object(light_obj)
    
    def on_show(self):
        """
        Called each time the view becomes active.
        """
        pass
    
    def on_hide(self):
        """
        Called when switching to a different view.
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

    # 2D drawing methods (forward to window if attached)
    # Allows drawing shapes and text in View3D.on_draw()
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
