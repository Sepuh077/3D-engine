"""
Window3D - Main application window for 3D rendering.
Similar to arcade.Window but for GPU-accelerated 3D.
"""
import time
import pygame
# Optional gfxdraw for anti-aliased 2D outlines (fallback to draw)
try:
    from pygame import gfxdraw
    HAS_GFXDRAW = True
except ImportError:
    HAS_GFXDRAW = False
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from .object3d import Object3D
from .camera import Camera3D
from .light import Light3D
from .color import Color, ColorType
from .keys import Keys
from src.physics import ColliderType, CollisionMode, Collider

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False
    print("Warning: ModernGL not installed. Install with: pip install moderngl")

if TYPE_CHECKING:
    from .view import View3D


@dataclass
class MeshGPU:
    key: object
    vbo: 'moderngl.Buffer'
    vao: 'moderngl.VertexArray'
    vertex_count: int
    ref_count: int = 0
    instance_vbo: Optional['moderngl.Buffer'] = None
    instance_capacity: int = 0
    instanced_vao: Optional['moderngl.VertexArray'] = None


@dataclass
class StaticBatch:
    vbo: 'moderngl.Buffer'
    vao: 'moderngl.VertexArray'
    vertex_count: int
    color: Tuple[float, float, float, float]
    center: np.ndarray
    radius: float


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
    in vec4 in_color;
    in vec2 in_uv;
    
    uniform mat4 mvp;
    uniform mat4 model;
    
    out vec3 frag_normal;
    out vec3 frag_position;
    out vec4 frag_v_color;
    out vec2 frag_uv;
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        frag_normal = mat3(model) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
        frag_v_color = in_color;
        frag_uv = in_uv;
    }
    '''

    VERTEX_SHADER_INSTANCED = '''
    #version 330 core

    in vec3 in_position;
    in vec3 in_normal;
    in vec4 in_color;
    in vec2 in_uv;
    in vec4 in_model_0;
    in vec4 in_model_1;
    in vec4 in_model_2;
    in vec4 in_model_3;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 frag_normal;
    out vec3 frag_position;
    out vec4 frag_v_color;
    out vec2 frag_uv;

    void main() {
        mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
        gl_Position = projection * view * model * vec4(in_position, 1.0);
        frag_normal = mat3(model) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
        frag_v_color = in_color;
        frag_uv = in_uv;
    }
    '''
    
    FRAGMENT_SHADER = '''
    #version 330 core
    
    in vec3 frag_normal;
    in vec3 frag_position;
    in vec4 frag_v_color;
    in vec2 frag_uv;
    
    uniform vec3 light_dir;
    uniform vec3 light_color;
    uniform float ambient;
    uniform vec4 base_color;
    uniform sampler2D tex;
    uniform bool use_texture;
    
    out vec4 frag_color;
    
    void main() {
        vec3 normal = normalize(frag_normal);
        vec3 light = normalize(-light_dir);
        
        // Two-sided lighting
        float diffuse = abs(dot(normal, light));
        
        // Combine vertex color and object tint
        vec4 albedo = frag_v_color * base_color;
        if (use_texture) {
            albedo *= texture(tex, frag_uv);
        }
        
        if (albedo.a < 0.001) discard;
        
        vec3 color = albedo.rgb * light_color * (ambient + diffuse * (1.0 - ambient));
        frag_color = vec4(color, albedo.a);
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

    # 2D overlay shaders for UI/shapes/text
    OVERLAY_VERTEX_SHADER = '''
    #version 330 core
    in vec2 in_pos;
    in vec2 in_tex;
    out vec2 frag_tex;
    void main() {
        gl_Position = vec4(in_pos, 0.0, 1.0);
        frag_tex = in_tex;
    }
    '''

    OVERLAY_FRAGMENT_SHADER = '''
    #version 330 core
    in vec2 frag_tex;
    uniform sampler2D tex;
    out vec4 frag_color;
    void main() {
        frag_color = texture(tex, frag_tex);
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
        self._ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Compile shaders
        self._program = self._ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self._instanced_program = self._ctx.program(
            vertex_shader=self.VERTEX_SHADER_INSTANCED,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self._collider_program = self._ctx.program(
            vertex_shader=self.COLLIDER_VERTEX_SHADER,
            fragment_shader=self.COLLIDER_FRAGMENT_SHADER,
        )

        # 2D overlay program for shapes and text
        self._overlay_program = self._ctx.program(
            vertex_shader=self.OVERLAY_VERTEX_SHADER,
            fragment_shader=self.OVERLAY_FRAGMENT_SHADER,
        )

        # GPU caches/batches
        self._mesh_cache = {}
        self._static_batches: List[StaticBatch] = []
        self._static_batches_active = False

        # Render options
        self.enable_instancing = True
        self.instancing_min = 2
        self.instancing_auto = True
        self.instancing_auto_min_objects = 64
        self.enable_culling = True
        self.culling_auto = True
        self.culling_auto_min_objects = 64

        # Simple uniform state cache
        self._last_base_color = None
        self._last_instanced_base_color = None

        # Simple profiler (caption-based)
        self.show_profiler = False
        self.profiler_interval = 0.25
        self._profiler_text = ""
        self._last_profiler_time = 0.0
        self._caption_base = title
        
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

        # 2D drawing setup: offscreen surface + OpenGL texture overlay
        # Supports shapes, text with fonts, drawn in on_draw()
        self._2d_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._fonts = {}
        self._image_cache = {}  # path -> loaded surface
        # Create full-screen quad for 2D overlay
        quad = np.array([
            -1.0, -1.0, 0.0, 1.0,  # bottom-left
             1.0, -1.0, 1.0, 1.0,  # bottom-right
             1.0,  1.0, 1.0, 0.0,  # top-right
            -1.0, -1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,  # top-left
        ], dtype=np.float32)
        self._2d_vbo = self._ctx.buffer(quad.tobytes())
        self._2d_vao = self._ctx.vertex_array(
            self._overlay_program,
            [(self._2d_vbo, '2f 2f', 'in_pos', 'in_tex')]
        )
        self._2d_texture = self._ctx.texture((width, height), 4)  # RGBA

        # Register as active window for global draw funcs (Arcade-style)
        from . import drawing
        drawing.set_window(self)

    
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
        self._ensure_mesh(obj)
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
            self._release_mesh(obj)
            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            self._release_mesh(obj)
        self.objects.clear()

    def move_object(self, obj: Object3D, delta: Tuple[float, float, float]) -> bool:
        """
        Move an object by delta.
        """
        # Check first collider's mode (IGNORE skips collision)
        coll = obj.get_component(Collider)
        if coll and coll.collision_mode == CollisionMode.IGNORE:
            return obj.move(*delta)
        return obj.move(*delta)

    def _resolve_collision(self, a: Object3D, b: Object3D, manifold):
        # Minimal depen + velocity project (slide, no jitter/vibrate on wall)
        depth = getattr(manifold, 'depth', 0.0)
        if depth <= 0:
            return
        push = depth + 1e-5
        normal = manifold.normal
        if a.static and b.static:
            return
        elif a.static:
            b._position -= normal * push
            # Project b vel: full stop if into, else slide
            dot = np.dot(b.velocity, normal)
            if dot < 0:
                b.velocity = np.zeros(3, dtype=np.float32)  # stay still
            else:
                b.velocity -= dot * normal
            b._mark_dirty()
            # Update colliders (moved from obj._update_cache)
            for c in b.get_components(Collider):
                c.update_bounds()
        elif b.static:
            a._position += normal * push
            # Project a vel: full stop if into, else slide
            dot = np.dot(a.velocity, normal)
            if dot < 0:
                a.velocity = np.zeros(3, dtype=np.float32)  # stay still
            else:
                a.velocity -= dot * normal
            a._mark_dirty()
            for c in a.get_components(Collider):
                c.update_bounds()
        else:
            a._position += normal * (push / 2)
            b._position -= normal * (push / 2)
            # Project vels: full stop if pushing into, else slide
            for obj in (a, b):
                dot = np.dot(obj.velocity, normal)
                if dot < 0:  # trying to move into wall
                    obj.velocity = np.zeros(3, dtype=np.float32)  # stay still
                else:
                    obj.velocity -= dot * normal  # allow slide
            a._mark_dirty()
            b._mark_dirty()
            for c in a.get_components(Collider):
                c.update_bounds()
            for c in b.get_components(Collider):
                c.update_bounds()

    def _process_collisions(self):
        # Loop over *all colliders* (multi-collider support; no obj level)
        all_cols = []
        for o in self._active_objects():
            all_cols.extend(o.get_components(Collider))
        if not all_cols:
            return

        from collections import defaultdict
        # Track collider pairs (per-collider _current_collisions for events)
        current_collisions = defaultdict(set)  # key: collider, value: set of other colliders
        from src.physics.collision import get_collision_manifold, objects_collide

        # Check non-statics vs all (masks replace groups/relations; check *all* collider pairs)
        for ca in all_cols:
            if ca.object3d.static or ca.collision_mode == CollisionMode.IGNORE:
                continue
            
            perform_final_check = True

            # Continuous sweep (per obj of collider)
            a = ca.object3d
            if ca.collision_mode == CollisionMode.CONTINUOUS:
                delta = a._position - a._prev_position
                speed = np.linalg.norm(delta)
                if speed > 1e-6:
                    a._position = np.copy(a._prev_position)
                    a._mark_dirty()
                    steps = max(1, int(speed / 0.1))
                    step = delta / steps
                    last_safe = np.copy(a._position)
                    
                    for _ in range(steps):
                        a._position += step
                        a._mark_dirty()
                        hit_solid = False
                        for cb in all_cols:
                            if cb is ca or cb.object3d is a:
                                continue
                            # Mask filter
                            if (ca.layer & cb.collision_mask) == 0 or (cb.layer & ca.collision_mask) == 0:
                                continue
                            if ca.check_collision(cb):
                                current_collisions[ca].add(cb)
                                current_collisions[cb].add(ca)
                                # Block if mode NORMAL/CONTINUOUS (solid); TRIGGER detect but pass
                                if ca.collision_mode in (CollisionMode.NORMAL, CollisionMode.CONTINUOUS):
                                    manifold = get_collision_manifold(ca, cb)
                                    if manifold:
                                        self._resolve_collision(a, cb.object3d, manifold)
                                    a.velocity = np.zeros(3, dtype=np.float32)
                                    hit_solid = True
                                    break
                        if hit_solid:
                            a._position = np.copy(last_safe)
                            a._mark_dirty()
                            break
                        last_safe = np.copy(a._position)
                    else:
                        perform_final_check = False
            
            # Normal snapshot
            if perform_final_check:
                for cb in all_cols:
                    if cb is ca or cb.object3d is a:
                        continue
                    # Mask filter
                    if (ca.layer & cb.collision_mask) == 0 or (cb.layer & ca.collision_mask) == 0:
                        continue
                    ca.update_bounds()
                    cb.update_bounds()
                    if objects_collide(ca, cb):
                        current_collisions[ca].add(cb)
                        current_collisions[cb].add(ca)
                        # Detect; block if mode NORMAL/CONTINUOUS; TRIGGER detect but pass
                        is_solid = ca.collision_mode in (CollisionMode.NORMAL, CollisionMode.CONTINUOUS)
                        if is_solid:
                            manifold = get_collision_manifold(ca, cb)
                            if manifold:
                                self._resolve_collision(a, cb.object3d, manifold)
        
        # Update collision events (per-collider _current_collisions)
        for c in all_cols:
            prev = c._current_collisions
            now = current_collisions.get(c, set())
            for oc in now - prev:
                c.OnCollisionEnter(oc)
            for oc in now & prev:
                c.OnCollisionStay(oc)
            for oc in prev - now:
                c.OnCollisionExit(oc)
            c._current_collisions = now.copy()
            
        # Update prev for next frame (continuous uses it)
        for obj in self._active_objects():
            obj._update_prev_position()

    def _update_profiler(self, stats: dict):
        if not self.show_profiler:
            return

        now = time.perf_counter()
        if now - self._last_profiler_time < self.profiler_interval:
            return

        self._last_profiler_time = now
        self._profiler_text = (
            f"objs {stats['visible']}/{stats['total']} "
            f"culled {stats['culled']} "
            f"inst {stats['instanced_objs']}x{stats['instanced_batches']} "
            f"single {stats['single_objs']} "
            f"static {stats['static_batches']} "
            f"{stats['cpu_ms']:.1f}ms"
        )
        self._apply_caption()

    def _active_objects(self) -> List[Object3D]:
        return self._current_view.objects if self._current_view else self.objects

    def _get_or_create_mesh(self, obj: Object3D) -> Optional[MeshGPU]:
        key = obj.get_mesh_key()
        if key is None:
            if not obj._gpu_initialized:
                obj._init_gpu(self._ctx, self._program)
            return None

        mesh = self._mesh_cache.get(key)
        if mesh is None:
            flat_vertices, flat_normals, flat_colors, flat_uvs = obj._get_flattened_geometry()
            
            if flat_vertices is None:
                raise RuntimeError("Object has no geometry loaded")

            vertex_data = np.hstack([flat_vertices, flat_normals, flat_colors, flat_uvs]).astype(np.float32)

            vbo = self._ctx.buffer(vertex_data.tobytes())
            vao = self._ctx.vertex_array(
                self._program,
                [(vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')]
            )

            mesh = MeshGPU(
                key=key,
                vbo=vbo,
                vao=vao,
                vertex_count=len(flat_vertices),
                ref_count=0,
            )
            self._mesh_cache[key] = mesh

        return mesh

    def _ensure_mesh(self, obj: Object3D):
        mesh = self._get_or_create_mesh(obj)
        if mesh is None:
            obj._mesh = None
            obj._gpu_initialized = True
            return

        if obj._mesh is mesh:
            return

        if obj._mesh is None and getattr(obj, "_vao", None) is not None:
            obj._release_gpu()

        if obj._mesh is not None:
            self._release_mesh(obj)

        mesh.ref_count += 1
        obj._mesh = mesh
        obj._gpu_initialized = True

    def _release_mesh(self, obj: Object3D):
        if obj._mesh is None:
            obj._release_gpu()
            return

        mesh = obj._mesh
        obj._mesh = None
        obj._gpu_initialized = False
        mesh.ref_count -= 1

        if mesh.ref_count <= 0:
            if mesh.instanced_vao:
                mesh.instanced_vao.release()
            if mesh.instance_vbo:
                mesh.instance_vbo.release()
            mesh.vao.release()
            mesh.vbo.release()
            self._mesh_cache.pop(mesh.key, None)

    def clear_static_batches(self):
        for batch in self._static_batches:
            batch.vao.release()
            batch.vbo.release()
        self._static_batches = []
        self._static_batches_active = False

    def build_static_batches(self):
        """
        Build GPU batches for static objects in the active scene.
        Call this after creating/moving static objects.
        """
        self.clear_static_batches()

        groups = defaultdict(list)
        for obj in self._active_objects():
            if not obj.visible or not obj.static:
                continue
            key = (obj.get_mesh_key(), tuple(obj._color))
            groups[key].append(obj)

        for (_, color), objs in groups.items():
            vertices_list = []
            normals_list = []
            colors_list = []
            uvs_list = []

            for obj in objs:
                flat_vertices, flat_normals, flat_colors, flat_uvs = obj._get_flattened_geometry()
                
                if flat_vertices is None:
                    continue

                model = obj.get_model_matrix()
                
                ones = np.ones((len(flat_vertices), 1), dtype=np.float32)
                v_h = np.hstack([flat_vertices, ones])
                v_world = v_h @ model

                m3 = model[:3, :3]
                try:
                    normal_mat = np.linalg.inv(m3)
                except np.linalg.LinAlgError:
                    normal_mat = np.eye(3, dtype=np.float32)
                n_world = flat_normals @ normal_mat
                norms = np.linalg.norm(n_world, axis=1, keepdims=True)
                n_world = n_world / np.maximum(norms, 1e-6)

                vertices_list.append(v_world[:, :3])
                normals_list.append(n_world)
                colors_list.append(flat_colors)
                uvs_list.append(flat_uvs)

            if not vertices_list:
                continue

            verts = np.vstack(vertices_list)
            norms = np.vstack(normals_list)
            cols = np.vstack(colors_list)
            uvs = np.vstack(uvs_list)
            
            vertex_data = np.hstack([verts, norms, cols, uvs]).astype(np.float32)

            vbo = self._ctx.buffer(vertex_data.tobytes())
            vao = self._ctx.vertex_array(
                self._program,
                [(vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')]
            )

            min_v = verts.min(axis=0)
            max_v = verts.max(axis=0)
            center = (min_v + max_v) * 0.5
            radius = float(np.linalg.norm(verts - center, axis=1).max())

            self._static_batches.append(
                StaticBatch(
                    vbo=vbo,
                    vao=vao,
                    vertex_count=len(verts),
                    color=color if len(color) == 4 else (*color, 1.0),
                    center=center,
                    radius=radius,
                )
            )

        self._static_batches_active = bool(self._static_batches)

    def _ensure_instanced_vao(self, mesh: MeshGPU, instance_count: int):
        if instance_count <= 0:
            return

        if mesh.instance_capacity < instance_count or mesh.instance_vbo is None:
            mesh.instance_capacity = max(instance_count, mesh.instance_capacity * 2, 16)
            mesh.instance_vbo = self._ctx.buffer(reserve=mesh.instance_capacity * 64)
            if mesh.instanced_vao:
                mesh.instanced_vao.release()
            mesh.instanced_vao = self._ctx.vertex_array(
                self._instanced_program,
                [
                    (mesh.vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv'),
                    (mesh.instance_vbo, '4f 4f 4f 4f /i',
                     'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
                ]
            )
    
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

        # Clear static batches when switching views
        if self._static_batches_active:
            self.clear_static_batches()
        
        # Attach new view
        self._current_view = view
        view._attach_window(self)
        
        # Initialize GPU for view's objects
        for obj in view.objects:
            if not obj._gpu_initialized:
                self._ensure_mesh(obj)
        
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
        self._caption_base = title
        self._apply_caption()

    def _apply_caption(self):
        title = self._caption_base
        if self.show_profiler and self._profiler_text:
            title = f"{title} | {self._profiler_text}"
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
        """Called when the window is resized. Recreates 2D overlay."""
        self.width = width
        self.height = height
        # Recreate 2D surface and texture
        self._2d_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        if hasattr(self, '_2d_texture'):
            self._2d_texture.release()
            self._2d_texture = self._ctx.texture((width, height), 4)
    
    # =========================================================================
    # 2D drawing (shapes, text, UI overlay)
    # =========================================================================
    
    def _get_font(self, font_name: Optional[str] = None, font_size: int = 24):
        """Get or create cached font."""
        key = (font_name or 'default', font_size)
        if key not in self._fonts:
            if font_name and (font_name.lower().endswith(('.ttf', '.otf')) or '/' in font_name or '\\' in font_name):
                # Load from file path
                self._fonts[key] = pygame.font.Font(font_name, font_size)
            else:
                # System font or default
                self._fonts[key] = pygame.font.SysFont(font_name, font_size)
        return self._fonts[key]
    
    def draw_text(self, text: str, x: int, y: int, color: ColorType = Color.WHITE,
                  font_size: int = 24, font_name: Optional[str] = None,
                  anchor_x: str = 'left', anchor_y: str = 'top',
                  baseline_adjust: bool = True) -> None:
        """Draw text at screen position (x,y top of text bounding box by default)."""
        font = self._get_font(font_name, font_size)
        # Convert color 0-1 to 0-255
        if len(color) == 3:
            rgb = tuple(int(c * 255) for c in color)
            alpha = 255
        else:
            rgb = tuple(int(c * 255) for c in color[:3])
            alpha = int(color[3] * 255)
        text_surf = font.render(text, True, rgb)  # antialias=True for quality
        # Handle alpha by creating alpha surface
        if alpha < 255:
            text_surf = text_surf.convert_alpha()
            # Simple alpha: multiply alpha channel (for uniform alpha)
            arr = pygame.surfarray.pixels_alpha(text_surf)
            arr[:] = (arr[:] * (alpha / 255)).astype(np.uint8)
            del arr  # release
        # Apply anchors (precise via bounding height)
        w, h = text_surf.get_size()
        if anchor_x == 'center':
            x -= w // 2
        elif anchor_x == 'right':
            x -= w
        if anchor_y == 'center':
            y -= h // 2
        elif anchor_y == 'bottom':
            y -= h
        # Optional baseline adjust to fix y-shifts across fonts/sizes
        if baseline_adjust:
            y -= font.get_ascent() // 6  # empirical top-align consistency
        self._2d_surface.blit(text_surf, (x, y))
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int,
                       color: ColorType, border_width: int = 0) -> None:
        """Draw filled or bordered rectangle. border_width=0 for fill."""
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self._2d_surface, col, rect, border_width)
    
    def draw_circle(self, x: int, y: int, radius: int, color: ColorType,
                    border_width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw filled (0) or bordered circle (AA if gfxdraw avail)."""
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        if border_width > 0 and HAS_GFXDRAW and aa:
            # Anti-aliased outline
            gfxdraw.aacircle(self._2d_surface, x, y, radius, col[:3])
            if border_width > 1:
                gfxdraw.aacircle(self._2d_surface, x, y, radius - 1, col[:3])  # thicker sim
        else:
            pygame.draw.circle(self._2d_surface, col, (x, y), radius, border_width)
    
    def draw_ellipse(self, x: int, y: int, width: int, height: int,
                     color: ColorType, border_width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw filled (0) or bordered ellipse/oval (AA if gfxdraw avail)."""
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        rect = pygame.Rect(x, y, width, height)
        if border_width > 0 and HAS_GFXDRAW and aa:
            # Anti-aliased outline (gfxdraw no width, sim with rect adjust)
            gfxdraw.aaellipse(self._2d_surface, x + width//2, y + height//2, width//2, height//2, col[:3])
        else:
            pygame.draw.ellipse(self._2d_surface, col, rect, border_width)
    
    def draw_polygon(self, points: List[Tuple[int, int]], color: ColorType,
                     border_width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw filled (0) or outlined polygon (AA if gfxdraw avail)."""
        if len(points) < 3:
            return  # invalid for polygon
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        if border_width > 0 and HAS_GFXDRAW and aa:
            gfxdraw.aapolygon(self._2d_surface, points, col[:3])
        else:
            pygame.draw.polygon(self._2d_surface, col, points, border_width)
    
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int],
                  color: ColorType, width: int = 2, aa: bool = True) -> None:  # thicker + AA default
        """Draw a line (AA via aaline if requested)."""
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        if aa:
            # Use built-in AA line (thin; gfxdraw.aaline may vary by pygame version)
            pygame.draw.aaline(self._2d_surface, col[:3], start, end)
        else:
            pygame.draw.line(self._2d_surface, col, start, end, width)
    
    def draw_image(self, image: Union[str, pygame.Surface], x: int, y: int,
                   scale: float = 1.0, alpha: float = 1.0) -> None:
        """Draw image (path str or Surface). Cached for paths; scale/alpha ok."""
        if isinstance(image, str):
            if image not in self._image_cache:
                surf = pygame.image.load(image).convert_alpha()
                self._image_cache[image] = surf
            surf = self._image_cache[image]
        else:
            surf = image
        if scale != 1.0:
            new_size = (int(surf.get_width() * scale), int(surf.get_height() * scale))
            surf = pygame.transform.scale(surf, new_size)
        if alpha < 1.0:
            surf = surf.copy()
            surf.set_alpha(int(alpha * 255))
        self._2d_surface.blit(surf, (x, y))
    
    def _render_2d_overlay(self):
        """Render 2D surface as textured quad on top of 3D scene."""
        # Upload surface to GPU texture (Pygame RGBA -> OpenGL)
        data = pygame.image.tostring(self._2d_surface, "RGBA", False)
        self._2d_texture.write(data)
        # Draw full-screen quad with blending, no depth
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._2d_texture.use(location=0)
        self._overlay_program['tex'].value = 0
        self._2d_vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.DEPTH_TEST)
    
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

        # Use first collider (normal case; supports multi via get_components)
        coll = obj.get_component(Collider)
        if not coll:
            return
        t = coll.type

        if t == ColliderType.CUBE:
            # Call collider methods (no world_* on Object3D)
            center, axes, extents = coll.get_world_obb() or (np.zeros(3), np.eye(3), np.ones(3))

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

        elif t == ColliderType.SPHERE:
            center, radius = coll.get_world_sphere() or (np.zeros(3), 1.0)

            model = np.eye(4, dtype=np.float32)
            model[:3, :3] *= radius
            model[3, :3] = center

            vao = self._sphere_vao

        elif t == ColliderType.CYLINDER:
            center, radius, half_h = coll.get_world_cylinder() or (np.zeros(3), 1.0, 1.0)
            _, axes, _ = coll.get_world_obb() or (np.zeros(3), np.eye(3), np.ones(3))

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
            
        elif t == ColliderType.MESH:
            # Fallback to drawing OBB for Mesh colliders
            center, axes, extents = coll.get_world_obb() or (np.zeros(3), np.eye(3), np.ones(3))

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

        mvp = model @ view @ proj
        self._collider_program['mvp'].write(mvp.tobytes())
        vao.render(moderngl.LINES)

    # =========================================================================
    # Rendering
    # =========================================================================
    
    def _render(self):
        """Render the scene with vertex colors OR real textures."""
        r, g, b = self.background_color
        self._ctx.clear(r, g, b)

        # Clear 2D overlay surface for new frame (draws happen in on_draw)
        self._2d_surface.fill((0, 0, 0, 0))

        # ------------------------------------------------------------
        # Camera / view setup
        # ------------------------------------------------------------
        if self._current_view:
            camera = self._current_view.camera
            light = self._current_view.light
            objects = self._current_view.objects
        else:
            camera = self.camera
            light = self.light
            objects = self.objects

        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.aspect)

        # ------------------------------------------------------------
        # Light uniforms
        # ------------------------------------------------------------
        for program in (self._program, self._instanced_program):
            program['light_dir'].value = tuple(light.direction)
            program['light_color'].value = tuple(light.color)
            program['ambient'].value = light.ambient

        # ------------------------------------------------------------
        # Visibility + culling + Sorting
        # ------------------------------------------------------------
        opaque_objects = []
        transparent_objects = []

        for obj in objects:
            if not obj.visible:
                continue
            self._ensure_mesh(obj)
            
            # Transparency check
            is_transparent = False
            if len(obj._color) == 4 and obj._color[3] < 0.99:
                is_transparent = True
            
            if is_transparent:
                transparent_objects.append(obj)
            else:
                opaque_objects.append(obj)

        # Sort transparent objects back-to-front
        if transparent_objects:
            cam_pos = camera.position
            transparent_objects.sort(key=lambda o: -np.linalg.norm(o.position - cam_pos))

        # ------------------------------------------------------------
        # Draw Helper
        # ------------------------------------------------------------
        def draw_objects(obj_list):
            for obj in obj_list:
                mesh = obj._mesh
                model = obj.get_model_matrix()
                mvp = model @ view @ projection

                self._program['mvp'].write(mvp.astype(np.float32).tobytes())
                self._program['model'].write(model.astype(np.float32).tobytes())

                # Texture path for GLTF etc (check on obj)
                use_texture = False
                if getattr(obj, "_uses_texture", False):
                    # Create GL texture once
                    if not hasattr(obj, "_gl_texture"):
                        tex_img = (obj._texture_image * 255).astype(np.uint8)
                        h, w = tex_img.shape[:2]
                        tex = self._ctx.texture((w, h), 4, tex_img.tobytes())
                        tex.build_mipmaps()
                        obj._gl_texture = tex

                    obj._gl_texture.use(location=0)
                    self._program['tex'].value = 0
                    use_texture = True

                self._program['use_texture'].value = use_texture

                # ============================================================
                # VERTEX COLOR / BASE COLOR PATH
                # ============================================================
                color = tuple(obj._color)
                rgba = color if len(color) == 4 else (*color, 1.0)
                self._program['base_color'].value = rgba

                if mesh is not None:
                    vao = mesh.vao
                    count = mesh.vertex_count
                else:
                    vao = obj._vao
                    count = None

                if count:
                    vao.render(moderngl.TRIANGLES, vertices=count)
                else:
                    vao.render(moderngl.TRIANGLES)

        # ------------------------------------------------------------
        # Draw Opaque (Depth Write ON)
        # ------------------------------------------------------------
        self._ctx.depth_mask = True
        draw_objects(opaque_objects)

        # ------------------------------------------------------------
        # Draw Transparent (Depth Write OFF)
        # ------------------------------------------------------------
        if transparent_objects:
            self._ctx.depth_mask = False
            draw_objects(transparent_objects)
            self._ctx.depth_mask = True  # Restore for next frame

        # ------------------------------------------------------------
        # Custom draw hooks
        # ------------------------------------------------------------
        if self._current_view:
            self._current_view.on_draw()
        self.on_draw()

        # Render 2D overlay on top (after all 3D and custom draws)
        self._render_2d_overlay()

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

        print(self.objects.__len__())
        
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
            
            # Auto collision detection + events + resolution (after user update)
            self._process_collisions()
            
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
        
        # Release 2D overlay resources
        if hasattr(self, '_2d_texture'):
            self._2d_texture.release()
        if hasattr(self, '_2d_vao'):
            self._2d_vao.release()
        if hasattr(self, '_2d_vbo'):
            self._2d_vbo.release()
        if hasattr(self, '_overlay_program'):
            self._overlay_program.release()
        
        self._program.release()
        self._ctx.release()
        pygame.quit()
