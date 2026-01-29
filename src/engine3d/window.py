"""
Window3D - Main application window for 3D rendering.
Similar to arcade.Window but for GPU-accelerated 3D.
"""
import pygame
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

from .object3d import Object3D
from .camera import Camera3D
from .light import Light3D
from .color import Color, ColorType
from .keys import Keys

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False
    print("Warning: ModernGL not installed. Install with: pip install moderngl")

if TYPE_CHECKING:
    from .view import View3D


class Window3D:
    """
    Main application window for 3D rendering.
    
    Similar to arcade.Window - subclass this to create your application.
    
    Example:
        class MyGame(Window3D):
            def setup(self):
                self.cube = self.load_object("cube.obj")
                self.cube.position = (0, 0, 0)
                
            def on_update(self, delta_time):
                self.cube.rotation_y += delta_time * 30
                
            def on_key_press(self, key, modifiers):
                if key == Keys.ESCAPE:
                    self.close()
        
        MyGame(800, 600, "My 3D Game").run()
    """
    
    # Shader source code
    VERTEX_SHADER = '''
    #version 330 core
    
    in vec3 in_position;
    in vec3 in_normal;
    
    uniform mat4 mvp;
    uniform mat4 model;
    
    out vec3 frag_normal;
    out vec3 frag_position;
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        frag_normal = mat3(model) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
    }
    '''
    
    FRAGMENT_SHADER = '''
    #version 330 core
    
    in vec3 frag_normal;
    in vec3 frag_position;
    
    uniform vec3 light_dir;
    uniform vec3 light_color;
    uniform float ambient;
    uniform vec3 base_color;
    
    out vec4 frag_color;
    
    void main() {
        vec3 normal = normalize(frag_normal);
        vec3 light = normalize(-light_dir);
        
        // Two-sided lighting
        float diffuse = abs(dot(normal, light));
        
        vec3 color = base_color * light_color * (ambient + diffuse * (1.0 - ambient));
        frag_color = vec4(color, 1.0);
    }
    '''

    COLLIDER_VERTEX_SHADER = '''
    #version 330 core

    in vec3 in_position;
    uniform mat4 mvp;

    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
    }
    '''

    COLLIDER_FRAGMENT_SHADER = '''
    #version 330 core

    uniform vec3 color;
    out vec4 frag_color;

    void main() {
        frag_color = vec4(color, 1.0);
    }
    '''
    
    def __init__(self, 
                 width: int = 800, 
                 height: int = 600, 
                 title: str = "3D Engine",
                 resizable: bool = False,
                 vsync: bool = True,
                 background_color: ColorType = (0.1, 0.1, 0.15)):
        """
        Initialize the window.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
            resizable: Allow window resizing
            vsync: Enable vertical sync
            background_color: Background color (RGB 0-1)
        """
        if not HAS_MODERNGL:
            raise ImportError("ModernGL is required. Install with: pip install moderngl")
        
        self.width = width
        self.height = height
        self.title = title
        self.background_color = background_color
        
        # Initialize pygame
        pygame.init()
        
        # Create OpenGL window
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if resizable:
            flags |= pygame.RESIZABLE
        
        pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption(title)
        
        # Create ModernGL context
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        
        # Compile shaders
        self._program = self._ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self._collider_program = self._ctx.program(
            vertex_shader=self.COLLIDER_VERTEX_SHADER,
            fragment_shader=self.COLLIDER_FRAGMENT_SHADER,
        )
        
        # Scene elements
        self.objects: List[Object3D] = []
        self.camera = Camera3D()
        self.light = Light3D()
        
        # View system
        self._current_view: Optional['View3D'] = None
        
        # Timing
        self._clock = pygame.time.Clock()
        self._running = False
        self._fps = 60
        self._delta_time = 0.0
        
        # Input state
        self._keys_pressed = set()
        self._mouse_position = (0, 0)
        self._mouse_buttons = set()
        
        # Setup done flag
        self._setup_done = False

        self._cube_vao = self._create_unit_cube_wire()
        self._sphere_vao = self._create_unit_sphere_wire(24)
        self._cylinder_vao = self._create_unit_cylinder_wire(24)

    
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
        
        # Initialize GPU resources
        obj._init_gpu(self._ctx, self._program)
        self.objects.append(obj)
        
        return obj
    
    def load_object(self, filename: str, **kwargs) -> Object3D:
        """
        Load and add a 3D object from file.
        
        Alias for add_object() with a filename.
        
        Args:
            filename: Path to OBJ file
            **kwargs: position, scale, color, etc.
            
        Returns:
            The loaded Object3D
        """
        return self.add_object(filename, **kwargs)
    
    def remove_object(self, obj: Object3D):
        """Remove object from scene."""
        if obj in self.objects:
            obj._release_gpu()
            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            obj._release_gpu()
        self.objects.clear()
    
    # =========================================================================
    # View management
    # =========================================================================
    
    def show_view(self, view: 'View3D'):
        """
        Switch to a different view.
        
        Args:
            view: The View3D to switch to
        """
        from .view import View3D
        
        # Detach current view
        if self._current_view:
            self._current_view._detach_window()
        
        # Attach new view
        self._current_view = view
        view._attach_window(self)
        
        # Initialize GPU for view's objects
        for obj in view.objects:
            if not obj._gpu_initialized:
                obj._init_gpu(self._ctx, self._program)
        
        view.on_show()
    
    @property
    def current_view(self) -> Optional['View3D']:
        """Get the current view."""
        return self._current_view
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._clock.get_fps()
    
    @property
    def delta_time(self) -> float:
        """Time since last frame in seconds."""
        return self._delta_time
    
    @property
    def size(self) -> Tuple[int, int]:
        """Window size as (width, height)."""
        return (self.width, self.height)
    
    @property
    def aspect(self) -> float:
        """Aspect ratio (width / height)."""
        return self.width / self.height
    
    def set_caption(self, title: str):
        """Set window title."""
        self.title = title
        pygame.display.set_caption(title)
    
    # =========================================================================
    # Input state
    # =========================================================================
    
    def is_key_pressed(self, key: int) -> bool:
        """Check if a key is currently pressed."""
        return key in self._keys_pressed
    
    def is_mouse_button_pressed(self, button: int) -> bool:
        """Check if a mouse button is currently pressed."""
        return button in self._mouse_buttons
    
    @property
    def mouse_position(self) -> Tuple[int, int]:
        """Current mouse position."""
        return self._mouse_position
    
    # =========================================================================
    # Lifecycle methods (override in subclass)
    # =========================================================================
    
    def setup(self):
        """
        Called once when the application starts.
        Override to set up your scene.
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
    # Input methods (override in subclass)
    # =========================================================================
    
    def on_key_press(self, key: int, modifiers: int):
        """Called when a key is pressed."""
        pass
    
    def on_key_release(self, key: int, modifiers: int):
        """Called when a key is released."""
        pass
    
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        """Called when a mouse button is pressed."""
        pass
    
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        """Called when a mouse button is released."""
        pass
    
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        """Called when the mouse moves."""
        pass
    
    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        """Called when the mouse wheel is scrolled."""
        pass
    
    def on_resize(self, width: int, height: int):
        """Called when the window is resized."""
        pass
    
    # =========================================================================
    # Collider debug drawing
    # =========================================================================
    def _create_unit_cube_wire(self):
        v = np.array([
            [-1,-1,-1],[ 1,-1,-1],
            [ 1,-1,-1],[ 1, 1,-1],
            [ 1, 1,-1],[-1, 1,-1],
            [-1, 1,-1],[-1,-1,-1],

            [-1,-1, 1],[ 1,-1, 1],
            [ 1,-1, 1],[ 1, 1, 1],
            [ 1, 1, 1],[-1, 1, 1],
            [-1, 1, 1],[-1,-1, 1],

            [-1,-1,-1],[-1,-1, 1],
            [ 1,-1,-1],[ 1,-1, 1],
            [ 1, 1,-1],[ 1, 1, 1],
            [-1, 1,-1],[-1, 1, 1],
        ], dtype=np.float32)

        vbo = self._ctx.buffer(v.tobytes())
        return self._ctx.vertex_array(
            self._collider_program,
            [(vbo, '3f', 'in_position')]
        )

    def _create_unit_sphere_wire(self, segments):
        verts = []
        angles = np.linspace(0, 2*np.pi, segments, endpoint=False)

        def ring(plane):
            nonlocal verts
            for i, a1 in enumerate(angles):
                a2 = angles[(i+1) % len(angles)]
                if plane == "xy":
                    p1 = (np.cos(a1), np.sin(a1), 0)
                    p2 = (np.cos(a2), np.sin(a2), 0)
                elif plane == "xz":
                    p1 = (np.cos(a1), 0, np.sin(a1))
                    p2 = (np.cos(a2), 0, np.sin(a2))
                else:  # yz
                    p1 = (0, np.cos(a1), np.sin(a1))
                    p2 = (0, np.cos(a2), np.sin(a2))
                verts += [p1, p2]

        ring("xy")
        ring("xz")
        ring("yz")

        v = np.array(verts, dtype=np.float32)
        vbo = self._ctx.buffer(v.tobytes())
        return self._ctx.vertex_array(
            self._collider_program,
            [(vbo, '3f', 'in_position')]
        )

    def _create_unit_cylinder_wire(self, segments):
        verts = []
        angles = np.linspace(0, 2*np.pi, segments, endpoint=False)

        for i, a1 in enumerate(angles):
            a2 = angles[(i+1) % len(angles)]

            x1, z1 = np.cos(a1), np.sin(a1)
            x2, z2 = np.cos(a2), np.sin(a2)

            # top ring (y=1)
            verts += [(x1,1,z1),(x2,1,z2)]
            # bottom ring (y=-1)
            verts += [(x1,-1,z1),(x2,-1,z2)]
            # vertical
            verts += [(x1,-1,z1),(x1,1,z1)]

        v = np.array(verts, dtype=np.float32)
        vbo = self._ctx.buffer(v.tobytes())
        return self._ctx.vertex_array(
            self._collider_program,
            [(vbo, '3f', 'in_position')]
        )

    def draw_collider(self, obj: Object3D, color=(0, 1, 0), line_width=1.0):
        camera = self._current_view.camera if self._current_view else self.camera
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(self.aspect)

        self._ctx.line_width = line_width
        self._collider_program['color'].value = tuple(color)

        t = obj.collider_type

        if t == "cube":
            center, axes, extents = obj.world_obb()

            S = np.array([
                [extents[0], 0, 0, 0],
                [0, extents[1], 0, 0],
                [0, 0, extents[2], 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

            R4 = np.eye(4, dtype=np.float32)
            R4[:3, :3] = axes

            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [center[0], center[1], center[2], 1],
            ], dtype=np.float32)

            model = S @ R4 @ T

            vao = self._cube_vao

        elif t == "sphere":
            center, radius = obj.world_sphere()

            model = np.eye(4, dtype=np.float32)
            model[:3, :3] *= radius
            model[3, :3] = center

            vao = self._sphere_vao

        elif t == "cylinder":
            center, radius, half_h = obj.world_cylinder()
            _, axes, _ = obj.world_obb()

            S = np.array([
                [radius, 0, 0, 0],
                [0, half_h, 0, 0],
                [0, 0, radius, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

            R4 = np.eye(4, dtype=np.float32)
            R4[:3, :3] = axes

            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [center[0], center[1], center[2], 1],
            ], dtype=np.float32)

            model = S @ R4 @ T

            vao = self._cylinder_vao

        mvp = model @ view @ proj
        self._collider_program['mvp'].write(mvp.tobytes())
        vao.render(moderngl.LINES)

    # =========================================================================
    # Rendering
    # =========================================================================
    
    def _render(self):
        """Render the scene."""
        # Clear screen
        r, g, b = self.background_color
        self._ctx.clear(r, g, b)
        
        # Get camera and light from current view or window
        if self._current_view:
            camera = self._current_view.camera
            light = self._current_view.light
            objects = self._current_view.objects
        else:
            camera = self.camera
            light = self.light
            objects = self.objects
        
        # Compute view and projection matrices
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.aspect)
        
        # Set light uniforms
        self._program['light_dir'].value = tuple(light.direction)
        self._program['light_color'].value = tuple(light.color)
        self._program['ambient'].value = light.ambient
        
        # Render each object
        for obj in objects:
            if not obj.visible:
                continue
            
            if not obj._gpu_initialized:
                obj._init_gpu(self._ctx, self._program)
            
            # Get model matrix
            model = obj.get_model_matrix()
            
            # Compute MVP
            mvp = model @ view @ projection
            
            # Set uniforms
            self._program['mvp'].write(mvp.astype(np.float32).tobytes())
            self._program['model'].write(model.astype(np.float32).tobytes())
            self._program['base_color'].value = tuple(obj._color)
            
            # Draw
            obj._vao.render(moderngl.TRIANGLES)
        
        # Call on_draw for custom rendering
        if self._current_view:
            self._current_view.on_draw()
        self.on_draw()
        
        # Swap buffers
        pygame.display.flip()
    
    # =========================================================================
    # Event handling
    # =========================================================================
    
    def _handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                
            elif event.type == pygame.KEYDOWN:
                self._keys_pressed.add(event.key)
                mods = pygame.key.get_mods()
                if self._current_view:
                    self._current_view.on_key_press(event.key, mods)
                self.on_key_press(event.key, mods)
                
            elif event.type == pygame.KEYUP:
                self._keys_pressed.discard(event.key)
                mods = pygame.key.get_mods()
                if self._current_view:
                    self._current_view.on_key_release(event.key, mods)
                self.on_key_release(event.key, mods)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._mouse_buttons.add(event.button)
                x, y = event.pos
                mods = pygame.key.get_mods()
                
                # Handle scroll wheel
                if event.button == 4:  # Scroll up
                    if self._current_view:
                        self._current_view.on_mouse_scroll(x, y, 0, 1)
                    self.on_mouse_scroll(x, y, 0, 1)
                elif event.button == 5:  # Scroll down
                    if self._current_view:
                        self._current_view.on_mouse_scroll(x, y, 0, -1)
                    self.on_mouse_scroll(x, y, 0, -1)
                else:
                    if self._current_view:
                        self._current_view.on_mouse_press(x, y, event.button, mods)
                    self.on_mouse_press(x, y, event.button, mods)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_buttons.discard(event.button)
                x, y = event.pos
                mods = pygame.key.get_mods()
                if self._current_view:
                    self._current_view.on_mouse_release(x, y, event.button, mods)
                self.on_mouse_release(x, y, event.button, mods)
                
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                dx, dy = event.rel
                self._mouse_position = (x, y)
                if self._current_view:
                    self._current_view.on_mouse_motion(x, y, dx, dy)
                self.on_mouse_motion(x, y, dx, dy)
                
            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self._ctx.viewport = (0, 0, event.w, event.h)
                if self._current_view:
                    self._current_view.on_resize(event.w, event.h)
                self.on_resize(event.w, event.h)
    
    # =========================================================================
    # Main loop
    # =========================================================================
    
    def run(self, fps: int = 60):
        """
        Start the main loop.
        
        Args:
            fps: Target frames per second (default 60)
        """
        self._fps = fps
        self._running = True
        
        # Call setup if not done
        if not self._setup_done:
            self.setup()
            self._setup_done = True
        
        # Main loop
        while self._running:
            # Calculate delta time
            self._delta_time = self._clock.tick(fps) / 1000.0
            
            # Handle events
            self._handle_events()
            
            # Update
            if self._current_view:
                self._current_view.on_update(self._delta_time)
            self.on_update(self._delta_time)
            
            # Render
            self._render()
            
            # Update title with FPS (optional)
            # pygame.display.set_caption(f"{self.title} - {self.fps:.1f} FPS")
        
        # Cleanup
        self._cleanup()
    
    def close(self):
        """Close the window and exit."""
        self._running = False
    
    def _cleanup(self):
        """Release all resources."""
        # Release GPU resources
        for obj in self.objects:
            obj._release_gpu()
        
        if self._current_view:
            for obj in self._current_view.objects:
                obj._release_gpu()
        
        self._program.release()
        self._ctx.release()
        pygame.quit()
