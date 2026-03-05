# Engine3D - Simple GPU-Accelerated 3D Engine for Python

A beginner-friendly 3D engine inspired by [Arcade](https://arcade.academy/), providing an easy-to-use API while leveraging GPU acceleration via ModernGL.

## Features

- 🎮 **Arcade-like API** - Simple, intuitive interface similar to arcade.Window
- 🚀 **GPU Accelerated** - Uses ModernGL for 100-1000x faster rendering than software 
- 📦 **Easy Object Loading** - Load OBJ files with one line of code
- 🎥 **Built-in Camera** - First-person and orbit camera modes
- 💡 **Lighting** - Directional lighting with ambient
- 🎭 **View System** - Easy scene management (menu, game, pause, etc.)
- 🎨 **Color Utilities** - Predefined colors and utilities

## Installation

```bash
pip install pygame moderngl numpy
```

## Quick Start

```python
from src.engine3d import Window3D, Keys, Color, Time

class MyGame(Window3D):
    def setup(self):
        # Load a 3D object
        self.cube = self.load_object("model.obj")
        self.cube.position = (0, 0, 0)
        self.cube.color = Color.ORANGE
        
        # Set up camera
        self.camera.position = (0, 5, 10)
        self.camera.look_at((0, 0, 0))
    
    def on_update(self):
        # Rotate the cube
        self.cube.rotation_y += 30 * Time.delta_time
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.close()

# Run the game
MyGame(800, 600, "My 3D Game").run()
```

## Examples

### Basic Example
```python
from src.engine3d import Window3D, Object3D, Keys, Color, Time

class BasicGame(Window3D):
    def setup(self):
        self.obj = self.load_object("example/stairs_modular_right.obj")
        self.camera.position = (0, 5, 15)
    
    def on_update(self):
        delta_time = Time.delta_time
        self.obj.rotation_y += 30 * delta_time
        
        # Camera controls
        if self.is_key_pressed(Keys.LEFT):
            self.camera.orbit(-delta_time, 0)
        if self.is_key_pressed(Keys.RIGHT):
            self.camera.orbit(delta_time, 0)
    
    def on_mouse_scroll(self, x, y, sx, sy):
        self.camera.zoom(-sy * 2)

BasicGame(800, 600, "Basic Example").run()
```

### 100 Objects at 60 FPS
```python
from src.engine3d import Window3D, Color

class ManyObjects(Window3D):
    def setup(self):
        for i in range(100):
            obj = self.load_object("model.obj")
            obj.position = (i % 10 * 3, 0, i // 10 * 3)
            obj.color = Color.random_bright()
        
        self.camera.position = (15, 20, 40)
        self.camera.look_at((15, 0, 15))
    
    def on_update(self):
        for obj in self.objects:
            obj.rotation_y += 30 * Time.delta_time

ManyObjects(800, 600, "100 Objects Demo").run()
```

### Using Scenes
```python
from src.engine3d import Window3D, Scene3D, Keys, Time

class MenuScene(Scene3D):
    def setup(self):
        self.title = self.load_object("title.obj")
        self.camera.position = (0, 5, 10)
    
    def on_key_press(self, key, mods):
        if key == Keys.ENTER:
            self.window.show_scene(GameScene())

class GameScene(Scene3D):
    def setup(self):
        self.player = self.load_object("player.obj")
    
    def on_update(self):
        if self.window.is_key_pressed(Keys.W):
            self.player.z -= 5 * Time.delta_time
    
    def on_key_press(self, key, mods):
        if key == Keys.ESCAPE:
            self.window.show_scene(MenuScene())

class Game(Window3D):
    def setup(self):
        self.show_scene(MenuScene())

Game(800, 600, "Game with Scenes").run()
```

## API Reference

### Window3D
Main application window.

```python
class Window3D:
    # Properties
    width: int
    height: int
    fps: float
    delta_time: float
    camera: Camera3D
    light: Light3D
    objects: List[Object3D]
    current_scene: Scene3D
    
    # Methods
    def setup(self)                              # Override: called once at start
    def on_update(self)                          # Override: called every frame
    def on_draw(self)                            # Override: custom drawing
    def on_key_press(self, key, modifiers)       # Override: key pressed
    def on_key_release(self, key, modifiers)     # Override: key released
    def on_mouse_press(self, x, y, button, mods) # Override: mouse pressed
    def on_mouse_motion(self, x, y, dx, dy)      # Override: mouse moved
    def on_mouse_scroll(self, x, y, sx, sy)      # Override: mouse scrolled
    
    def load_object(filename, **kwargs) -> Object3D
    def add_object(obj: Object3D) -> Object3D
    def remove_object(obj: Object3D)
    def show_scene(scene: Scene3D)
    def is_key_pressed(key: int) -> bool
    def run(fps: int = 60)
    def close()
```

### Object3D
A 3D object in the scene.

```python
class Object3D:
    # Position
    position: tuple   # (x, y, z)
    x, y, z: float
    
    # Rotation (degrees)
    rotation: tuple   # (rx, ry, rz)
    rotation_x, rotation_y, rotation_z: float
    
    # Scale
    scale: float              # Uniform scale
    scale_xyz: tuple          # Non-uniform (sx, sy, sz)
    
    # Appearance
    color: tuple              # RGB (0-1)
    visible: bool
    static: bool             # Static objects can be batched for speed
    
    # Methods
    def load(filename: str)
    def move(dx, dy, dz)
    def rotate(dx, dy, dz)    # Degrees
    def show()
    def hide()
```

For large scenes, mark objects as `static = True` and call
`Window3D.build_static_batches()` after setup to merge static geometry.

### Performance helpers
`Window3D` enables frustum culling and instancing by default. You can tweak:

```python
window.enable_culling = True
window.enable_instancing = True
window.instancing_min = 2  # minimum objects per mesh+color before instancing
window.instancing_auto = True
window.instancing_auto_min_objects = 64
window.culling_auto = True
window.culling_auto_min_objects = 64
```

Enable a lightweight profiler (updates the window caption):

```python
window.show_profiler = True
window.profiler_interval = 0.25  # seconds
```

### Camera3D
Camera with position, target, and projection.

```python
class Camera3D:
    position: tuple
    target: tuple
    x, y, z: float
    fov: float
    near, far: float
    
    def look_at(target: tuple)
    def move(dx, dy, dz)
    def move_forward(distance)
    def move_right(distance)
    def move_up(distance)
    def orbit(yaw, pitch)     # Radians
    def zoom(delta)
```

### Light3D
Directional light.

```python
class Light3D:
    direction: tuple          # Normalized direction vector
    color: tuple             # RGB (0-1)
    intensity: float
    ambient: float
    
    def point_from(position, target)
```

### Color
Predefined colors and utilities.

```python
class Color:
    # Predefined
    WHITE, BLACK, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA
    GRAY, DARK_GRAY, LIGHT_GRAY
    ORANGE, PINK, PURPLE, BROWN, GOLD, SILVER
    SKY_BLUE, FOREST_GREEN, OCEAN_BLUE, SAND
    
    # Methods
    @staticmethod def from_rgb(r, g, b) -> tuple      # 0-255
    @staticmethod def from_hex(hex_str) -> tuple      # "#FF5500"
    @staticmethod def random() -> tuple
    @staticmethod def random_bright() -> tuple
    @staticmethod def lerp(c1, c2, t) -> tuple
```

### Keys
Keyboard constants.

```python
class Keys:
    A-Z, KEY_0-KEY_9, F1-F12
    UP, DOWN, LEFT, RIGHT
    SPACE, ENTER, ESCAPE, TAB, BACKSPACE, DELETE
    LSHIFT, RSHIFT, LCTRL, RCTRL, LALT, RALT
```

## Project Structure

```
3d-engine/
├── src/
│   └── engine3d/
│       ├── __init__.py     # Main imports
│       ├── window.py       # Window3D class
│       ├── scene.py        # Scene3D class
│       ├── object3d.py     # Object3D class
│       ├── camera.py       # Camera3D class
│       ├── light.py        # Light3D class
│       ├── color.py        # Color utilities
│       └── keys.py         # Key constants
├── examples/
│   ├── example_basic.py
│   ├── example_many_objects.py
│   ├── example_scenes.py
│   └── example_fps_camera.py
└── example/
    └── stairs_modular_right.obj
```

## Performance

| Configuration | FPS with 100 Objects |
|---------------|----------------------|
| Software (pygame) | ~0.3 FPS |
| Engine3D (GPU) | 200+ FPS |

## License

MIT License
