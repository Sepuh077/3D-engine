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

from .gameobject import GameObject
from .object3d import Object3D
from .graphics.material import UnlitMaterial, LitMaterial, SpecularMaterial, EmissiveMaterial, TransparentMaterial
from .camera import Camera3D
from .light import Light3D, DirectionalLight3D
from .graphics.color import Color, ColorType
from .input.keys import Keys
from src.physics import ColliderType, CollisionMode, CollisionRelation, Collider
from src.physics.rigidbody import Rigidbody
from .component import Script, Time

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False
    print("Warning: ModernGL not installed. Install with: pip install moderngl")

if TYPE_CHECKING:
    from .scene import Scene3D


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
                
            def on_update(self):
                self.cube.rotation_y += Time.delta_time * 30
                
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
    
    #define MAX_POINT_LIGHTS 4
    uniform int num_point_lights;
    uniform vec3 point_light_positions[MAX_POINT_LIGHTS];
    uniform vec3 point_light_colors[MAX_POINT_LIGHTS];
    uniform float point_light_intensities[MAX_POINT_LIGHTS];
    uniform float point_light_ranges[MAX_POINT_LIGHTS];
    
    uniform vec4 base_color;
    uniform sampler2D tex;
    uniform bool use_texture;

    // Material properties
    uniform int material_type; // 0: Unlit, 1: Lit, 2: Specular, 3: Emissive
    uniform vec3 specular_color;
    uniform float shininess;
    uniform float emissive_intensity;
    uniform vec3 view_pos;
    
    out vec4 frag_color;
    
    void main() {
        vec3 normal = normalize(frag_normal);
        vec3 view_dir = normalize(view_pos - frag_position);
        
        // Combine vertex color and object tint
        vec4 albedo = frag_v_color * base_color;
        if (use_texture) {
            albedo *= texture(tex, frag_uv);
        }
        
        if (albedo.a < 0.001) discard;

        vec3 result_color;

        if (material_type == 0) { // Unlit
            result_color = albedo.rgb;
        } 
        else if (material_type == 3) { // Emissive
            result_color = albedo.rgb * emissive_intensity;
        }
        else { // Lit or Specular
            // Directional light
            vec3 dir_light_dir = normalize(-light_dir);
            float dir_diffuse = max(dot(normal, dir_light_dir), 0.0);
            vec3 diffuse_light = light_color * (ambient + dir_diffuse * (1.0 - ambient));
            
            vec3 specular_light = vec3(0.0);
            if (material_type == 2) { // Specular
                vec3 reflect_dir = reflect(-dir_light_dir, normal);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
                specular_light += light_color * spec * specular_color;
            }

            // Point lights
            for (int i = 0; i < num_point_lights; ++i) {
                vec3 light_vec = point_light_positions[i] - frag_position;
                float distance = length(light_vec);
                if (distance < point_light_ranges[i]) {
                    vec3 pl_dir = normalize(light_vec);
                    float pl_diffuse = max(dot(normal, pl_dir), 0.0);
                    
                    // Attenuation (quadratic mix)
                    float attenuation = 1.0 - (distance / point_light_ranges[i]);
                    attenuation = attenuation * attenuation; // Smooth falloff
                    
                    diffuse_light += point_light_colors[i] * pl_diffuse * point_light_intensities[i] * attenuation;

                    if (material_type == 2) { // Specular
                        vec3 reflect_dir = reflect(-pl_dir, normal);
                        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
                        specular_light += point_light_colors[i] * spec * specular_color * point_light_intensities[i] * attenuation;
                    }
                }
            }
            result_color = albedo.rgb * diffuse_light + specular_light;
        }
        
        frag_color = vec4(result_color, albedo.a);
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
                 background_color: ColorType = (0.1, 0.1, 0.15),
                 use_pygame_window: bool = True,
                 use_pygame_events: bool = True):
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
        # When embedded, SDL_WINDOWID is set by the host.
        
        self._use_pygame_window = use_pygame_window
        self._use_pygame_events = use_pygame_events and use_pygame_window

        pygame.init()

        if self._use_pygame_window:
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            if resizable:
                flags |= pygame.RESIZABLE

            pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption(title)
            pygame.event.set_allowed([
                pygame.QUIT,
                pygame.KEYDOWN,
                pygame.KEYUP,
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP,
                pygame.MOUSEMOTION,
                pygame.VIDEORESIZE,
            ])

        # Create ModernGL context
        try:
            self._ctx = moderngl.create_context(require=330)
        except Exception:
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
        self.objects: List[GameObject] = []
        
        # Default camera for window-only mode
        self._camera_go = GameObject("Default Camera")
        self.camera = Camera3D()
        self._camera_go.add_component(self.camera)
        self._camera_go.transform.position = (0, 5, 10)
        self._camera_go.transform.look_at((0, 0, 0))

        # Editor overlay options
        self.show_editor_overlays = False
        self.editor_selected_object: Optional[GameObject] = None
        self.editor_show_camera = True
        self.editor_show_axis = True
        self.editor_show_gizmo = True
        self.active_camera_override: Optional[Camera3D] = None

        # Scene system
        self._current_scene: Optional['Scene3D'] = None
        
        # Timing
        self._clock = pygame.time.Clock()
        self._running = False
        self._fps = 60
        self._delta_time = 0.0
        Time.delta_time = 0.0
        
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

    @property
    def light(self) -> Optional[DirectionalLight3D]:
        """Get the first DirectionalLight3D component in the window, or None if none exists."""
        for obj in self.objects:
            l = obj.get_component(DirectionalLight3D)
            if l:
                return l
        return None

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
        
        # Initialize GPU resources for the MeshRenderer part
        obj3d_comp = go.get_component(Object3D)
        if obj3d_comp:
            self._ensure_mesh(obj3d_comp)
        self.objects.append(go)
        
        # Start scripts on the new object
        go.start_scripts()
        
        return go

    def load_object(self, filename: str, **kwargs) -> GameObject:
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
    
    def remove_object(self, obj: GameObject):
        """Remove object from scene."""
        if obj in self.objects:
            if obj.get_component(Object3D): self._release_mesh(obj.get_component(Object3D))
            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            if obj.get_component(Object3D): self._release_mesh(obj.get_component(Object3D))
        self.objects.clear()

    def move_object(self, obj: GameObject, delta: Tuple[float, float, float]) -> bool:
        """
        Move an object by delta.
        """
        # Check first collider's mode (IGNORE skips collision)
        coll = obj.get_component(Collider)
        if coll and coll.collision_mode == CollisionMode.IGNORE:
            return obj.transform.move(*delta)
        return obj.transform.move(*delta)

    def _resolve_collision(self, a: GameObject, b: GameObject, manifold):
        # Minimal depen + velocity project (slide, no jitter/vibrate on wall)
        depth = getattr(manifold, 'depth', 0.0)
        if depth <= 0:
            return
        push = depth + 1e-5
        normal = manifold.normal
        a_static = a.get_component(Rigidbody) and a.get_component(Rigidbody).is_static
        b_static = b.get_component(Rigidbody) and b.get_component(Rigidbody).is_static
        
        if a_static and b_static:
            return
        elif a_static:
            b.transform._local_position -= normal * push
            # Project b vel: full stop if into, else slide
            if b.get_component(Rigidbody):
                dot = np.dot(b.get_component(Rigidbody).velocity, normal)
                if dot < 0:
                    b.get_component(Rigidbody).velocity -= dot * normal
            b.transform._mark_dirty()
            # Update colliders (moved from obj._update_cache)
            for c in b.get_components(Collider):
                c.update_bounds()
        elif b_static:
            a.transform._local_position += normal * push
            # Project a vel: full stop if into, else slide
            if a.get_component(Rigidbody):
                dot = np.dot(a.get_component(Rigidbody).velocity, normal)
                if dot < 0:
                    a.get_component(Rigidbody).velocity -= dot * normal
            a.transform._mark_dirty()
            for c in a.get_components(Collider):
                c.update_bounds()
        else:
            a.transform._local_position += normal * (push / 2)
            b.transform._local_position -= normal * (push / 2)
            # Project vels: full stop if pushing into, else slide
            for obj in (a, b):
                if not obj.get_component(Rigidbody):
                    continue
                dot = np.dot(obj.get_component(Rigidbody).velocity, normal)
                if dot < 0:  # trying to move into wall
                    obj.get_component(Rigidbody).velocity -= dot * normal  # allow slide
            a.transform._mark_dirty()
            b.transform._mark_dirty()
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

        # Check non-statics vs all (use ColliderGroup for relations; *all* pairs)
        for ca in all_cols:
            if (ca.game_object.get_component(Rigidbody) and ca.game_object.get_component(Rigidbody).is_static) or ca.collision_mode == CollisionMode.IGNORE:
                continue
            
            perform_final_check = True

            # Continuous sweep (per obj of collider)
            a = ca.game_object
            if ca.collision_mode == CollisionMode.CONTINUOUS:
                delta = a.transform._local_position - a.transform._prev_position
                speed = np.linalg.norm(delta)
                if speed > 1e-6:
                    a.transform._local_position = np.copy(a.transform._prev_position)
                    a.transform._mark_dirty()
                    steps = max(1, int(speed / 0.1))
                    step = delta / steps
                    last_safe = np.copy(a.transform._local_position)
                    
                    for _ in range(steps):
                        a.transform._local_position += step
                        a.transform._mark_dirty()
                        hit_solid = False
                        for cb in all_cols:
                            if cb is ca or cb.game_object is a:
                                continue
                            # ColliderGroup: IGNORE skip; TRIGGER detect/pass; SOLID block
                            relation = ca.group.get_relation(cb.group)
                            if relation == CollisionRelation.IGNORE:
                                continue
                            if ca.check_collision(cb):
                                current_collisions[ca].add(cb)
                                current_collisions[cb].add(ca)
                                # block only on SOLID (TRIGGER passes)
                                if relation == CollisionRelation.SOLID:
                                    manifold = get_collision_manifold(ca, cb)
                                    if manifold:
                                        self._resolve_collision(a, cb.game_object, manifold)
                                    if a.get_component(Rigidbody):
                                        a.get_component(Rigidbody).velocity = np.zeros(3, dtype=np.float32)
                                    hit_solid = True
                                    break
                        if hit_solid:
                            a.transform._local_position = np.copy(last_safe)
                            a.transform._mark_dirty()
                            break
                        last_safe = np.copy(a.transform._local_position)
                    else:
                        perform_final_check = False
            
            # Normal snapshot
            if perform_final_check:
                for cb in all_cols:
                    if cb is ca or cb.game_object is a:
                        continue
                    # ColliderGroup relation: IGNORE skip, TRIGGER detect/pass, SOLID block
                    relation = ca.group.get_relation(cb.group)
                    if relation == CollisionRelation.IGNORE:
                        continue
                    ca.update_bounds()
                    cb.update_bounds()
                    if objects_collide(ca, cb):
                        current_collisions[ca].add(cb)
                        current_collisions[cb].add(ca)
                        # block only on SOLID (Normal can't pass); TRIGGER pass thru
                        if relation == CollisionRelation.SOLID:
                            manifold = get_collision_manifold(ca, cb)
                            if manifold:
                                self._resolve_collision(a, cb.game_object, manifold)
        
        # Update collision events (per-collider _current_collisions)
        for c in all_cols:
            prev = c._current_collisions
            now = current_collisions.get(c, set())
            for oc in now - prev:
                c.OnCollisionEnter(oc)
                # Propagate to scripts on the same game object
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_enter(oc)
            for oc in now & prev:
                c.OnCollisionStay(oc)
                # Propagate to scripts on the same game object
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_stay(oc)
            for oc in prev - now:
                c.OnCollisionExit(oc)
                # Propagate to scripts on the same game object
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_exit(oc)
            c._current_collisions = now.copy()
            
        # Update prev for next frame (continuous uses it)
        for obj in self._active_objects():
            obj.transform._update_prev_position()

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

    def _active_objects(self) -> List[GameObject]:
        return self._current_scene.objects if self._current_scene else self.objects

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
            if not obj.get_component(Object3D) or not obj.get_component(Object3D)._visible or not (obj.get_component(Rigidbody) and obj.get_component(Rigidbody).is_static):
                continue
            key = (obj.get_component(Object3D).get_mesh_key(), tuple(obj.get_component(Object3D)._color))
            groups[key].append(obj)

        for (_, color), objs in groups.items():
            vertices_list = []
            normals_list = []
            colors_list = []
            uvs_list = []

            for obj in objs:
                flat_vertices, flat_normals, flat_colors, flat_uvs = obj.get_component(Object3D)._get_flattened_geometry()
                
                if flat_vertices is None:
                    continue

                model = obj.transform.get_model_matrix()
                
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
    # Scene management
    # =========================================================================
    
    def show_scene(self, scene: 'Scene3D'):
        """
        Switch to a different scene.
        
        Args:
            scene: The Scene3D to switch to
        """
        
        # Detach current scene
        if self._current_scene:
            self._current_scene._detach_window()

        # Clear static batches when switching scenes
        if self._static_batches_active:
            self.clear_static_batches()
        
        # Attach new scene
        self._current_scene = scene
        scene._attach_window(self)
        
        # Initialize GPU for scene's objects
        for obj in scene.objects:
            obj3d = obj.get_component(Object3D)
            if obj3d and not obj3d._gpu_initialized:
                self._ensure_mesh(obj3d)
        
        scene.on_show()
    
    @property
    def current_scene(self) -> Optional['Scene3D']:
        """Get the current scene."""
        return self._current_scene
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._clock.get_fps()
    
    @property
    def delta_time(self) -> float:
        """Unscaled time since last frame in seconds."""
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

    def project_point(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        """
        Project a 3D world position to screen space.

        Returns (x, y, depth) in screen pixels, or None if behind camera.
        """
        camera = self._current_scene.camera if self._current_scene else self.camera
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.aspect)
        vec = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)
        clip = vec @ view @ projection
        w = clip[3]
        if w <= 0.0:
            return None
        ndc = clip[:3] / w
        x = int((ndc[0] + 1.0) * 0.5 * self.width)
        y = int((1.0 - ndc[1]) * 0.5 * self.height)
        return (x, y, ndc[2])
    
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
        # Add default directional light
        light_obj = GameObject("Directional Light")
        light_obj.add_component(DirectionalLight3D())
        light_obj.transform.rotation = (-45, 30, 0)
        self.add_object(light_obj)
    
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
        self.width = max(1, width)
        self.height = max(1, height)
        # Recreate 2D surface and texture
        self._2d_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if hasattr(self, '_2d_texture'):
            self._2d_texture.release()
            self._2d_texture = self._ctx.texture((self.width, self.height), 4)
        if hasattr(self, '_ctx'):
            self._ctx.viewport = (0, 0, self.width, self.height)

    def bind_context(self):
        """Ensure this window's GL context is active before rendering."""
        if self._use_pygame_window:
            return
        try:
            self._ctx.use()
        except Exception:
            pass
    
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
        # Draw UI elements from current scene's canvas
        if self._current_scene:
            self._current_scene.canvas.draw(self._2d_surface)
        
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

    def draw_collider(self, obj: GameObject, color=(0, 1, 0), line_width=1.0):
        camera = self._current_scene.camera if self._current_scene else self.camera
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(self.aspect)

        self._ctx.line_width = line_width
        self._collider_program['color'].value = tuple(color)

        # Use first collider (normal case; supports multi via get_components)
        for coll in obj.get_components(Collider):
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
        
        # Use custom screen FBO if set (e.g. for Qt embedding where FBO changes)
        if getattr(self, '_screen_fbo', None):
            self._screen_fbo.clear(r, g, b)
            self._screen_fbo.use()
        else:
            self._ctx.clear(r, g, b)

        self.bind_context()

        # Clear 2D overlay surface for new frame (draws happen in on_draw)
        self._2d_surface.fill((0, 0, 0, 0))

        camera = self.active_camera_override
        if not camera:
            camera = self._current_scene.camera if self._current_scene else self.camera

        if not camera:
            return

        if self._current_scene:
            light = self._current_scene.light
            objects = self._current_scene.objects
        else:
            light = self.light
            objects = self.objects

        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.aspect)

        # ------------------------------------------------------------
        # Light uniforms
        # ------------------------------------------------------------
        from .light import PointLight3D
        point_lights = []
        for obj in objects:
            pls = obj.get_components(PointLight3D)
            point_lights.extend(pls)
            
        num_pl = min(len(point_lights), 4)

        for program in (self._program, self._instanced_program):
            program['view_pos'].value = tuple(camera.position)
            if light:
                # Multiply by intensity so the intensity property actually works
                l_col = (
                    light.color[0] * light.intensity,
                    light.color[1] * light.intensity,
                    light.color[2] * light.intensity
                )
                program['light_dir'].value = tuple(light.direction)
                program['light_color'].value = l_col
                program['ambient'].value = light.ambient
            else:
                program['light_dir'].value = (0.0, -1.0, 0.0)
                program['light_color'].value = (0.0, 0.0, 0.0)
                program['ambient'].value = 0.0
            
            if 'num_point_lights' in program:
                program['num_point_lights'].value = num_pl
                
                if num_pl > 0:
                    pos_vals = []
                    col_vals = []
                    int_vals = []
                    range_vals = []
                    for i in range(4):
                        if i < num_pl:
                            pl = point_lights[i]
                            # Use world_position if available
                            pos = pl.game_object.transform.world_position if pl.game_object else pl.position
                            pos_vals.extend(pos)
                            col_vals.extend(pl.color[:3] if len(pl.color) >= 3 else (1.0, 1.0, 1.0))
                            int_vals.append(float(pl.intensity))
                            range_vals.append(float(pl.range))
                        else:
                            pos_vals.extend([0.0, 0.0, 0.0])
                            col_vals.extend([0.0, 0.0, 0.0])
                            int_vals.append(0.0)
                            range_vals.append(0.0)
                    
                    if 'point_light_positions' in program:
                        program['point_light_positions'].write(np.array(pos_vals, dtype='f4').tobytes())
                    if 'point_light_colors' in program:
                        program['point_light_colors'].write(np.array(col_vals, dtype='f4').tobytes())
                    if 'point_light_intensities' in program:
                        program['point_light_intensities'].write(np.array(int_vals, dtype='f4').tobytes())
                    if 'point_light_ranges' in program:
                        program['point_light_ranges'].write(np.array(range_vals, dtype='f4').tobytes())

        # ------------------------------------------------------------
        # Visibility + culling + Sorting
        # ------------------------------------------------------------
        opaque_objects = []
        transparent_objects = []

        for obj in objects:
            obj3d = obj.get_component(Object3D)
            if not obj3d or not obj3d._visible:
                continue
            self._ensure_mesh(obj3d)
            
            # Transparency check
            is_transparent = False
            if isinstance(obj3d.material, TransparentMaterial) or obj3d.material.alpha < 0.99:
                is_transparent = True
            elif len(obj3d.material.color_vec4) == 4 and obj3d.material.color_vec4[3] < 0.99:
                is_transparent = True
            
            if is_transparent:
                transparent_objects.append((obj, obj3d))
            else:
                opaque_objects.append((obj, obj3d))

        # Sort transparent objects back-to-front
        if transparent_objects:
            cam_pos = camera.position
            transparent_objects.sort(key=lambda item: -np.linalg.norm(item[0].transform.position - cam_pos))

        # ------------------------------------------------------------
        # Draw Helper
        # ------------------------------------------------------------
        def draw_objects(obj_list):
            for go, obj3d in obj_list:
                mesh = obj3d._mesh
                model = go.transform.get_model_matrix()
                mvp = model @ view @ projection

                self._program['mvp'].write(mvp.astype(np.float32).tobytes())
                self._program['model'].write(model.astype(np.float32).tobytes())

                use_texture = False
                if getattr(obj3d, "_uses_texture", False):
                    if not hasattr(obj3d, "_gl_texture"):
                        tex_img = (obj3d._texture_image * 255).astype(np.uint8)
                        h, w = tex_img.shape[:2]
                        tex = self._ctx.texture((w, h), 4, tex_img.tobytes())
                        tex.build_mipmaps()
                        obj3d._gl_texture = tex

                    obj3d._gl_texture.use(location=0)
                    self._program['tex'].value = 0
                    use_texture = True

                self._program['use_texture'].value = use_texture

                # Material uniforms
                mat = obj3d.material
                if isinstance(mat, UnlitMaterial):
                    self._program['material_type'].value = 0
                elif isinstance(mat, LitMaterial):
                    self._program['material_type'].value = 1
                elif isinstance(mat, SpecularMaterial):
                    self._program['material_type'].value = 2
                    self._program['specular_color'].value = tuple(mat.specular_vec3)
                    self._program['shininess'].value = float(mat.shininess)
                elif isinstance(mat, EmissiveMaterial):
                    self._program['material_type'].value = 3
                    self._program['emissive_intensity'].value = float(mat.intensity)
                elif isinstance(mat, TransparentMaterial):
                    self._program['material_type'].value = 1 # Transparent uses Lit logic but with alpha
                else:
                    self._program['material_type'].value = 1 # Default to Lit

                rgba = tuple(mat.color_vec4)
                self._program['base_color'].value = rgba

                if mesh is not None:
                    vao = mesh.vao
                    count = mesh.vertex_count
                else:
                    vao = obj3d._vao
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
        if self._current_scene:
            self._current_scene.on_draw()
        self.on_draw()

        if self.show_editor_overlays:
            self._draw_editor_overlays()

        # Render 2D overlay on top (after all 3D and custom draws)
        self._render_2d_overlay()

        if self._use_pygame_window:
            pygame.display.flip()

    
    def _draw_editor_overlays(self):
        active_camera = self.active_camera_override or (self._current_scene.camera if self._current_scene else self.camera)
        
        if self.editor_show_axis:
            self._draw_editor_axis(active_camera)
            
        if self.editor_show_camera and self._current_scene:
            for obj in self._current_scene.objects:
                for cam in obj.get_components(Camera3D):
                    if cam != active_camera:
                        self._draw_editor_camera(cam)
                        
        if self.editor_show_gizmo and self.editor_selected_object:
            self._draw_editor_gizmo(self.editor_selected_object)
        self._draw_editor_colliders()

    def _draw_editor_colliders(self):
        for obj in self._active_objects():
            if obj.get_components(Collider):
                self.draw_collider(obj, color=(1.0, 0.0, 0.0), line_width=1.5)

    def _draw_editor_axis(self, camera: Camera3D):
        origin = (0.0, 0.0, 0.0)
        axis_length = 1.5
        axis_points = {
            "X": (axis_length, 0.0, 0.0),
            "Y": (0.0, axis_length, 0.0),
            "Z": (0.0, 0.0, axis_length),
        }
        axis_colors = {
            "X": (1.0, 0.3, 0.3),
            "Y": (0.3, 1.0, 0.3),
            "Z": (0.3, 0.3, 1.0),
        }
        origin_screen = self.project_point(origin)
        if not origin_screen:
            return
        for axis, target in axis_points.items():
            target_screen = self.project_point(target)
            if not target_screen:
                continue
            self.draw_line(origin_screen[:2], target_screen[:2], axis_colors[axis], width=2, aa=True)
            label_pos = (target_screen[0] + 6, target_screen[1] + 4)
            self.draw_text(axis, label_pos[0], label_pos[1], axis_colors[axis], font_size=14)

    def _draw_editor_camera(self, camera: Camera3D):
        cam_go = camera.game_object
        if not cam_go: return
        cam_pos = cam_go.transform.world_position
        origin = self.project_point(cam_pos)
        if not origin:
            return
        
        color = (1.0, 1.0, 1.0) # White
        self.draw_circle(origin[0], origin[1], 6, color, border_width=2, aa=True)
        self.draw_text("Camera", origin[0] + 8, origin[1] - 12, color, font_size=14)

        forward = cam_go.transform.forward
        right = cam_go.transform.right
        up = cam_go.transform.up

        near = camera.near
        far = min(camera.far, 20.0) # Show a reasonable frustum extent
        fov_rad = np.radians(camera.fov)
        aspect = self.aspect
        
        # Calculate half-height and half-width at near and far planes
        half_height_near = np.tan(fov_rad * 0.5) * near
        half_width_near = half_height_near * aspect
        half_height_far = np.tan(fov_rad * 0.5) * far
        half_width_far = half_height_far * aspect

        near_center = cam_pos + forward * near
        far_center = cam_pos + forward * far

        # Near plane corners (correct aspect ratio)
        near_corners = [
            near_center + right * half_width_near + up * half_height_near,  # top-right
            near_center - right * half_width_near + up * half_height_near,  # top-left
            near_center - right * half_width_near - up * half_height_near,  # bottom-left
            near_center + right * half_width_near - up * half_height_near,  # bottom-right
        ]
        # Far plane corners
        far_corners = [
            far_center + right * half_width_far + up * half_height_far,     # top-right
            far_center - right * half_width_far + up * half_height_far,     # top-left
            far_center - right * half_width_far - up * half_height_far,     # bottom-left
            far_center + right * half_width_far - up * half_height_far,     # bottom-right
        ]

        # Draw near plane rectangle
        for i in range(4):
            n0 = self.project_point(tuple(near_corners[i]))
            n1 = self.project_point(tuple(near_corners[(i + 1) % 4]))
            if n0 and n1:
                self.draw_line(n0[:2], n1[:2], color, width=2, aa=True)
        
        # Draw far plane rectangle
        for i in range(4):
            f0 = self.project_point(tuple(far_corners[i]))
            f1 = self.project_point(tuple(far_corners[(i + 1) % 4]))
            if f0 and f1:
                self.draw_line(f0[:2], f1[:2], color, width=1, aa=True)
        
        # Draw connecting lines from near to far (frustum edges)
        for i in range(4):
            n0 = self.project_point(tuple(near_corners[i]))
            f0 = self.project_point(tuple(far_corners[i]))
            if n0 and f0:
                self.draw_line(n0[:2], f0[:2], color, width=1, aa=True)
        
        # Draw forward direction line from camera position
        forward_point = cam_pos + forward * (near + 0.5)
        forward_screen = self.project_point(tuple(forward_point))
        if forward_screen:
            self.draw_line(origin[:2], forward_screen[:2], color, width=2, aa=True)

    def _draw_editor_gizmo(self, obj: GameObject):
        origin = self.project_point(tuple(obj.transform.world_position))
        if not origin:
            return
        gizmo_length = 1.0
        axes = {
            "X": obj.transform.right,
            "Y": obj.transform.up,
            "Z": obj.transform.forward,
        }
        colors = {
            "X": (1.0, 0.2, 0.2),
            "Y": (0.2, 1.0, 0.2),
            "Z": (0.2, 0.6, 1.0),
        }
        for axis, direction in axes.items():
            endpoint = obj.transform.world_position + direction * gizmo_length
            end_screen = self.project_point(tuple(endpoint))
            if not end_screen:
                continue
            self.draw_line(origin[:2], end_screen[:2], colors[axis], width=3, aa=True)
            self.draw_circle(end_screen[0], end_screen[1], 4, colors[axis], border_width=0, aa=True)
            self.draw_text(axis, end_screen[0] + 6, end_screen[1] + 4, colors[axis], font_size=12)

    # =========================================================================
    # Event handling
    # =========================================================================
    
    def _handle_events(self):
        """Process pygame events."""
        if not self._use_pygame_events:
            return
        for event in pygame.event.get():
            # Pass to scene's canvas UI first
            if self._current_scene and self._current_scene.canvas.process_pygame_event(event):
                continue  # UI handled this event
            
            if event.type == pygame.QUIT:
                self._running = False
                
            elif event.type == pygame.KEYDOWN:
                self._keys_pressed.add(event.key)
                mods = pygame.key.get_mods()
                if self._current_scene:
                    self._current_scene.on_key_press(event.key, mods)
                self.on_key_press(event.key, mods)
                
            elif event.type == pygame.KEYUP:
                self._keys_pressed.discard(event.key)
                mods = pygame.key.get_mods()
                if self._current_scene:
                    self._current_scene.on_key_release(event.key, mods)
                self.on_key_release(event.key, mods)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._mouse_buttons.add(event.button)
                x, y = event.pos
                mods = pygame.key.get_mods()
                
                # Handle scroll wheel
                if event.button == 4:  # Scroll up
                    if self._current_scene:
                        self._current_scene.on_mouse_scroll(x, y, 0, 1)
                    self.on_mouse_scroll(x, y, 0, 1)
                elif event.button == 5:  # Scroll down
                    if self._current_scene:
                        self._current_scene.on_mouse_scroll(x, y, 0, -1)
                    self.on_mouse_scroll(x, y, 0, -1)
                else:
                    if self._current_scene:
                        self._current_scene.on_mouse_press(x, y, event.button, mods)
                    self.on_mouse_press(x, y, event.button, mods)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_buttons.discard(event.button)
                x, y = event.pos
                mods = pygame.key.get_mods()
                if self._current_scene:
                    self._current_scene.on_mouse_release(x, y, event.button, mods)
                self.on_mouse_release(x, y, event.button, mods)
                
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                dx, dy = event.rel
                self._mouse_position = (x, y)
                if self._current_scene:
                    self._current_scene.on_mouse_motion(x, y, dx, dy)
                self.on_mouse_motion(x, y, dx, dy)
                
            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self._ctx.viewport = (0, 0, event.w, event.h)
                if self._current_scene:
                    self._current_scene.on_resize(event.w, event.h)
                self.on_resize(event.w, event.h)
    
    # =========================================================================
    # Main loop
    # =========================================================================

    def start(self, fps: int = 60):
        """Initialize the window for manual ticking or run()."""
        self._fps = fps
        if not self._setup_done:
            self.setup()
            self._setup_done = True

        for obj in self._active_objects():
            obj.start_scripts()

        self._running = True

    def tick(self, fps: Optional[int] = None, simulate: bool = True) -> bool:
        """Advance one frame. Returns False when closed."""
        if fps is not None:
            self._fps = fps
        if not self._running:
            self.start(self._fps)

        raw_dt = self._clock.tick(self._fps) / 1000.0
        self._delta_time = raw_dt
        Time.delta_time = raw_dt * Time.scale

        self._handle_events()

        if simulate:
            if self._current_scene:
                self._current_scene.on_update()
            self.on_update()

            for obj in self._active_objects():
                obj.update()

            if self._current_scene:
                self._current_scene.canvas.update(self._delta_time)

            self._process_collisions()

            for obj in self._active_objects():
                obj.update_end_of_frame()

        self._render()

        return self._running

    def run(self, fps: int = 60):
        """
        Start the main loop.
        
        Args:
            fps: Target frames per second (default 60)
        """
        self.start(fps)

        while self._running:
            self.tick(fps)

        self._cleanup()

    def close(self):
        """Close the window and exit."""
        self._running = False
    
    def _cleanup(self):
        """Release all resources."""
        # Release GPU resources
        for obj in self.objects:
            if obj.get_component(Object3D): obj.get_component(Object3D)._release_gpu()
        
        if self._current_scene:
            for obj in self._current_scene.objects:
                if obj.get_component(Object3D): obj.get_component(Object3D)._release_gpu()
        
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
