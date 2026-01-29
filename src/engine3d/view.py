"""
View3D - A scene/view that can be shown in a Window3D.
Similar to arcade.View.
"""
from typing import List, Optional, TYPE_CHECKING

from .object3d import Object3D
from .camera import Camera3D
from .light import Light3D

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
                
            def on_update(self, delta_time):
                self.player.rotation_y += delta_time * 30
                
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
        self.objects: List[Object3D] = []
        self.camera = Camera3D()
        self.light = Light3D()
        self._setup_done = False
    
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
    
    def add_object(self, obj_or_filename, **kwargs) -> Object3D:
        """
        Add a 3D object to the scene.
        
        Args:
            obj_or_filename: Object3D instance or path to OBJ file
            **kwargs: Passed to Object3D constructor if loading from file
            
        Returns:
            The added Object3D
        """
        if isinstance(obj_or_filename, Object3D):
            obj = obj_or_filename
        else:
            obj = Object3D(obj_or_filename, **kwargs)
        
        self.objects.append(obj)
        
        # Initialize GPU if window is available
        if self.window and self.window._ctx:
            self.window._ensure_mesh(obj)
        
        return obj
    
    def remove_object(self, obj: Object3D):
        """Remove object from scene."""
        if obj in self.objects:
            if self.window:
                self.window._release_mesh(obj)
            else:
                obj._release_gpu()
            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            if self.window:
                self.window._release_mesh(obj)
            else:
                obj._release_gpu()
        self.objects.clear()
    
    def get_objects_by_name(self, name: str) -> List[Object3D]:
        """Get all objects with a specific name."""
        return [obj for obj in self.objects if obj.name == name]
    
    def get_objects_by_tag(self, tag: str) -> List[Object3D]:
        """Get all objects with a specific tag."""
        return [obj for obj in self.objects if obj.tag == tag]
    
    def load_object(self, filename: str, **kwargs) -> Object3D:
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
        pass
    
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
    
    def on_update(self, delta_time: float):
        """
        Called every frame to update the scene.
        
        Args:
            delta_time: Time since last frame in seconds
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
