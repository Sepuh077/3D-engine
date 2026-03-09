"""
Engine3D - A simple, GPU-accelerated 3D engine for Python.

Similar to arcade's API, but for 3D graphics.

Example:
    from src.engine3d import Window3D, Object3D
    
    class MyGame(Window3D):
        def setup(self):
            self.player = self.load_object("player.obj")
            self.player.position = (0, 0, 0)
        
        def on_update(self):
            self.player.rotation_y += Time.delta_time
        
        def on_key_press(self, key, modifiers):
            if key == Keys.ESCAPE:
                self.close()
    
    MyGame(800, 600, "My Game").run()
"""

from .window import Window3D
from .scene import Scene3D
from .gameobject import GameObject
from .editor import run_editor
from .component import Component, Script, WaitForSeconds, WaitEndOfFrame, Time
from .transform import Transform
from src.physics.rigidbody import Rigidbody
from .object3d import Object3D, create_cube, create_sphere, create_plane
from .camera import Camera3D
from .light import Light3D, DirectionalLight3D, PointLight3D
from .input.keys import Keys
from .graphics.color import Color
from .particle import (
    ParticleSystem,
    ParticleBurst,
    linear_size_over_lifetime,
    linear_color_over_lifetime,
    linear_velocity_over_lifetime,
    SphereShape,
    ConeShape,
    BoxShape,
)

# UI System
from .ui import (
    UILayer,
    UIElement,
    UIContainer,
    UIEvent,
    Label,
    Button,
    CheckBox,
    Slider,
    ProgressBar,
    Panel,
    UIManager,
)

# Arcade-style global 2D drawing (separate module)
from .drawing import (
    get_window,
    draw_text,
    draw_rectangle,
    draw_circle,
    draw_ellipse,
    draw_polygon,
    draw_line,
    draw_image,
)

__all__ = [
    'Window3D',
    'Scene3D', 
    'GameObject',
    'run_editor',
    'Component',
    'Script',
    'WaitForSeconds',
    'WaitEndOfFrame',
    'Time',
    'Transform',
    'Rigidbody',
    'Object3D',
    'create_cube',
    'create_sphere',
    'create_plane',
    'Camera3D',
    'Light3D',
    'DirectionalLight3D',
    'PointLight3D',
    'Keys',
    'Color',
    'ParticleSystem',
    'ParticleBurst',
    'linear_size_over_lifetime',
    'linear_color_over_lifetime',
    'linear_velocity_over_lifetime',
    'SphereShape',
    'ConeShape',
    'BoxShape',
    # UI System
    'UILayer',
    'UIElement',
    'UIContainer',
    'UIEvent',
    'Label',
    'Button',
    'CheckBox',
    'Slider',
    'ProgressBar',
    'Panel',
    'UIManager',
    # Global 2D drawing (Arcade-style)
    'get_window',
    'draw_text',
    'draw_rectangle',
    'draw_circle',
    'draw_ellipse',
    'draw_polygon',
    'draw_line',
    'draw_image',
]

__version__ = '0.1.0'
