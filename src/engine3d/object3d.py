"Object3D - 3D object that can be loaded, positioned, rotated, and scaled."
import hashlib
import numpy as np
import trimesh
from typing import Tuple, Optional, List, TYPE_CHECKING

from src.physics import Component, Collider, Rigidbody
from trimesh.visual.texture import TextureVisuals
from .color import ColorType, Color

if TYPE_CHECKING:
    import moderngl
    from .window import Window3D


class Object3D:
    def __init__(
        self,
        filename: Optional[str] = None,
        position=(0, 0, 0),
        scale: float = 1.0,
        color: Optional[ColorType] = None,
    ):
        # ---------------- Transform ----------------
        self._position = np.array(position, dtype=np.float32)
        self._rotation = np.zeros(3, dtype=np.float32)
        self._scale = np.array([scale, scale, scale], dtype=np.float32)

        self._transform_dirty = True

        # Cached transforms
        self._cached_rotation = None
        self._cached_model = None
        
        # ---------------- Geometry ----------------
        self.mesh: Optional[trimesh.Trimesh] = None

        self._local_min = None
        self._local_max = None
        self._local_radius = None

        # Texture support (for GLTF etc)
        self._uses_texture = False
        self._texture_image = None
        self._uv = None

        # ---------------- Misc ----------------
        c = color if color is not None else (1, 1, 1)
        c = np.array(c, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            c = np.append(c, 1.0)
        self._color = c
        self._visible = True
        self._static = False
        self.name = "Object3D"
        self.tag = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self._prev_position = np.copy(self._position)

        # GPU handles (initialized later)
        self._vbo = None
        self._vao = None
        self._gpu_initialized = False

        # Mesh identity for batching/instancing
        self._mesh_key = None
        self._mesh = None

        # Components (user adds Collider/Rigidbody etc separately)
        self.components: List[Component] = []

        if filename:
            self.load(filename)

    # ======================================================================
    # Loading & geometry preprocessing (ONCE)
    # ======================================================================

    def load(self, filename: str):
        # Load with trimesh for full support including colors
        self._load_with_trimesh(filename)
        # Post-processing
        self._post_process_geometry(filename)

    def _post_process_geometry(self, geometry_name: str):
        if self.mesh is not None and len(self.mesh.vertices) > 0:
            # Center geometry
            center = self.mesh.vertices.mean(axis=0)
            self.mesh.apply_translation(-center)

            self._local_min = self.mesh.vertices.min(axis=0)
            self._local_max = self.mesh.vertices.max(axis=0)
            self._local_radius = np.linalg.norm(self.mesh.vertices, axis=1).max()

            # Ensure normals (trimesh computes lazily on .vertex_normals access)
            _ = self.mesh.vertex_normals
            self._mesh_key = ("geom", geometry_name)
            self._transform_dirty = True

    def _load_with_trimesh(self, filename: str):
        loaded = trimesh.load(filename)

        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"No geometry found in {filename}")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        self.mesh = mesh

        self._uses_texture = False
        self._texture_image = None
        self._uv = None

        if not hasattr(mesh, "visual"):
            return

        visual = mesh.visual

        # Vertex colors path (unified normalize/pad)
        if (
            hasattr(visual, "vertex_colors")
            and visual.vertex_colors is not None
            and len(visual.vertex_colors) == len(mesh.vertices)
        ):
            colors = visual.vertex_colors.astype(np.float32)
            if colors.max() > 1.0:
                colors /= 255.0
            if colors.shape[1] == 3:
                colors = np.pad(colors, ((0, 0), (0, 1)), constant_values=1.0)
            mesh.visual.vertex_colors = colors
            return

        # Textured path
        if isinstance(visual, TextureVisuals):
            material = visual.material
            alpha_mode = getattr(material, "alphaMode", "OPAQUE")
            if alpha_mode != "OPAQUE":
                self._uses_texture = True
                raw_img = (
                    getattr(material, "baseColorTexture", None)
                    or getattr(material, "image", None)
                )
                img_arr = np.asarray(raw_img)
                if img_arr.ndim == 2:
                    img_arr = np.stack([img_arr] * 3, axis=2)
                img_arr = img_arr.astype(np.float32) / 255.0
                if img_arr.shape[2] == 3:
                    img_arr = np.pad(img_arr, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
                self._texture_image = img_arr
                self._uv = visual.uv
                return

            # Texture-to-vertex-colors fallback (simplified sampling)
            img = (
                getattr(material, "baseColorTexture", None)
                or getattr(material, "image", None)
            )
            uv = visual.uv
            if img is None or uv is None:
                return

            img = np.array(img).astype(np.float32) / 255.0
            h, w = img.shape[:2]

            def sample_color(u, v):
                u = u % 1.0
                v = v % 1.0
                x = np.clip(int(u * (w - 1)), 0, w - 1)
                y = np.clip(int((1 - v) * (h - 1)), 0, h - 1)
                c = img[y, x]
                if c.shape[0] == 3:
                    c = np.append(c, 1.0)
                return c

            num_uv = len(uv)
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)

            v_colors = np.zeros((num_vertices, 4), dtype=np.float32)
            counts = np.zeros(num_vertices, dtype=np.int32)

            if num_uv == num_vertices:
                for vert_idx in range(num_vertices):
                    u, v = uv[vert_idx]
                    v_colors[vert_idx] = sample_color(u, v)
            elif num_uv == num_faces * 3:
                for face_idx, face in enumerate(mesh.faces):
                    for corner in range(3):
                        vert_idx = face[corner]
                        uv_idx = face_idx * 3 + corner
                        u, v = uv[uv_idx]
                        v_colors[vert_idx] += sample_color(u, v)
                        counts[vert_idx] += 1
                for i in range(num_vertices):
                    if counts[i] > 0:
                        v_colors[i] /= counts[i]
                    else:
                        v_colors[i] = [1, 1, 1, 1]
            else:
                return

            mesh.visual.vertex_colors = v_colors
            return

        # Simple material fallback
        material = getattr(visual, 'material', None)
        if material is not None:
            base = getattr(material, 'baseColor', None) or getattr(material, 'diffuse', [1.0, 1.0, 1.0, 1.0])
            base = np.array(base, dtype=np.float32)
            if len(base) == 3:
                base = np.append(base, 1.0)
            colors = np.full((len(mesh.vertices), 4), base, dtype=np.float32)
            mesh.visual.vertex_colors = colors

    # Dirty flag helper (marks transform dirty + colliders for runtime update)
    def _mark_dirty(self):
        self._transform_dirty = True
        # Mark colliders dirty (for lazy bounds update)
        for comp in self.components:
            if isinstance(comp, Collider):
                comp._transform_dirty = True

    # Component system (add/get like Unity)
    def add_component(self, component: Component) -> Component:
        component.object3d = self
        component.on_attach()
        self.components.append(component)
        return component

    def get_component(self, component_type: type) -> Optional[Component]:
        for comp in self.components:
            if isinstance(comp, component_type):
                return comp
        return None

    def get_components(self, component_type: type) -> List[Component]:
        return [comp for comp in self.components if isinstance(comp, component_type)]

    # =========================================================================
    # Position properties
    # =========================================================================
    @property
    def vertices(self):
        # Delegate to trimesh
        return self.mesh.vertices if self.mesh is not None else None

    @vertices.setter
    def vertices(self, v):
        # Update trimesh mesh
        if self.mesh is None:
            self.mesh = trimesh.Trimesh(vertices=v)
        else:
            self.mesh.vertices = v
        self._local_min = self.mesh.vertices.min(axis=0)
        self._local_max = self.mesh.vertices.max(axis=0)
        self._local_radius = np.linalg.norm(self.mesh.vertices, axis=1).max()
        self._mesh_key = None
    
    @property
    def position(self) -> np.ndarray:
        """Get position as tuple."""
        return self._position.copy()
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set position."""
        self._update_prev_position()
        self._position = np.array(value, dtype=np.float32)
        self._mark_dirty()
    
    @property
    def x(self) -> float:
        return float(self._position[0])
    
    @x.setter
    def x(self, value: float):
        """Set x."""
        self.position = (value, self.y, self.z)
    
    @property
    def y(self) -> float:
        return float(self._position[1])
    
    @y.setter
    def y(self, value: float):
        """Set y."""
        self.position = (self.x, value, self.z)
    
    @property
    def z(self) -> float:
        return float(self._position[2])
    
    @z.setter
    def z(self, value: float):
        """Set z."""
        self.position = (self.x, self.y, value)

    # ---------------- Bounds ----------------
    @property
    def min_x(self) -> float:
        """World-space minimum X coordinate."""
        self._update_cache()
        return float(self.collider.aabb[0][0])

    @min_x.setter
    def min_x(self, value: float):
        self.x += value - self.min_x

    @property
    def max_x(self) -> float:
        """World-space maximum X coordinate."""
        self._update_cache()
        return float(self.collider.aabb[1][0])

    @max_x.setter
    def max_x(self, value: float):
        self.x += value - self.max_x

    @property
    def min_y(self) -> float:
        """World-space minimum Y coordinate."""
        self._update_cache()
        return float(self.collider.aabb[0][1])

    @min_y.setter
    def min_y(self, value: float):
        self.y += value - self.min_y

    @property
    def max_y(self) -> float:
        """World-space maximum Y coordinate."""
        self._update_cache()
        return float(self.collider.aabb[1][1])

    @max_y.setter
    def max_y(self, value: float):
        self.y += value - self.max_y

    @property
    def min_z(self) -> float:
        """World-space minimum Z coordinate."""
        self._update_cache()
        return float(self.collider.aabb[0][2])

    @min_z.setter
    def min_z(self, value: float):
        self.z += value - self.min_z

    @property
    def max_z(self) -> float:
        """World-space maximum Z coordinate."""
        self._update_cache()
        return float(self.collider.aabb[1][2])

    @max_z.setter
    def max_z(self, value: float):
        self.z += value - self.max_z
    
    def move(self, dx: float = 0, dy: float = 0, dz: float = 0) -> bool:
        self._update_prev_position()
        delta = np.array([dx, dy, dz], dtype=np.float32)
        self._position += delta
        self._mark_dirty()
        self.velocity = delta
        return True

    def _update_prev_position(self):
        self._prev_position = np.copy(self._position)

    # =========================================================================
    # Rotation properties
    # =========================================================================
    
    @property
    def rotation(self) -> Tuple[float, float, float]:
        """Get rotation as tuple (degrees)."""
        return tuple(np.degrees(self._rotation))
    
    @rotation.setter
    def rotation(self, value: Tuple[float, float, float]):
        """Set rotation (degrees)."""
        self._rotation = np.radians(value).astype(np.float32)
        self._mark_dirty()
    
    @property
    def rotation_x(self) -> float:
        """Rotation around X axis in degrees."""
        return float(np.degrees(self._rotation[0]))
    
    @rotation_x.setter
    def rotation_x(self, value: float):
        self._rotation[0] = np.radians(value)
        self._mark_dirty()
    
    @property
    def rotation_y(self) -> float:
        """Rotation around Y axis in degrees."""
        return float(np.degrees(self._rotation[1]))
    
    @rotation_y.setter
    def rotation_y(self, value: float):
        self._rotation[1] = np.radians(value)
        self._mark_dirty()
    
    @property
    def rotation_z(self) -> float:
        """Rotation around Z axis in degrees."""
        return float(np.degrees(self._rotation[2]))
    
    @rotation_z.setter
    def rotation_z(self, value: float):
        self._rotation[2] = np.radians(value)
        self._mark_dirty()
    
    def rotate(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Rotate object by offset (degrees)."""
        self._rotation += np.radians([dx, dy, dz]).astype(np.float32)
        self._mark_dirty()
    
    # =========================================================================
    # Scale properties
    # =========================================================================
    
    @property
    def scale(self) -> float:
        """Get uniform scale."""
        return float(self._scale[0])
    
    @scale.setter
    def scale(self, value: float):
        """Set uniform scale."""
        self._scale = np.array([value, value, value], dtype=np.float32)
        self._mark_dirty()
    
    @property
    def scale_xyz(self) -> Tuple[float, float, float]:
        """Get non-uniform scale."""
        return tuple(self._scale)
    
    @scale_xyz.setter
    def scale_xyz(self, value: Tuple[float, float, float]):
        """Set non-uniform scale."""
        self._scale = np.array(value, dtype=np.float32)
        self._mark_dirty()
    
    # =========================================================================
    # Appearance properties
    # =========================================================================
    
    @property
    def color(self) -> Tuple[float, float, float]:
        """Get color as tuple."""
        return tuple(self._color)
    
    @color.setter
    def color(self, value: ColorType):
        """Set color."""
        # Normalize (support 0-1 or 0-255)
        c = np.array(value, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            c = np.append(c, 1.0)
        self._color = c
    
    @property
    def visible(self) -> bool:
        """Is object visible?"""
        return self._visible
    
    @visible.setter
    def visible(self, value: bool):
        """Set visibility."""
        self._visible = value

    @property
    def static(self) -> bool:
        """Is object static? Static objects can be batched/instanced."""
        return self._static

    @static.setter
    def static(self, value: bool):
        """Set static flag."""
        self._static = bool(value)
    
    def show(self):
        """Make object visible."""
        self._visible = True
    
    def hide(self):
        """Make object invisible."""
        self._visible = False

    # Mesh identity for caching/instancing
    def get_mesh_key(self):
        if self._mesh_key is not None:
            return self._mesh_key

        if self.mesh is None:
            return None

        h = hashlib.blake2b(digest_size=16)
        h.update(self.mesh.vertices.tobytes())
        h.update(self.mesh.faces.tobytes())
        self._mesh_key = ("geom", h.hexdigest())
        return self._mesh_key
    
    # =========================================================================
    # Model matrix (for rendering; no collider)
    # =========================================================================
    
    def get_model_matrix(self) -> np.ndarray:
        if not self._transform_dirty:
            return self._cached_model
        # Rotation matrix
        cx, cy, cz = np.cos(self._rotation)
        sx, sy, sz = np.sin(self._rotation)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rx @ Ry @ Rz
        self._cached_rotation = R
        # Model
        sx, sy, sz = self._scale
        tx, ty, tz = self._position
        S = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]], dtype=np.float32)
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = R
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [tx, ty, tz, 1]], dtype=np.float32)
        self._cached_model = S @ R4 @ T
        self._transform_dirty = False
        return self._cached_model
    
    # GPU helper for moderngl rendering
    def _get_flattened_geometry(self):
        if self.mesh is None:
            return None, None, None, None

        # Flatten using trimesh data (ensures correct colors from visual)
        faces = self.mesh.faces
        flat_vertices = self.mesh.vertices[faces.flatten()]
        flat_normals = self.mesh.vertex_normals[faces.flatten()]
        
        # Get colors from trimesh visual (fixed extraction + normalize 0-255->0-1)
        visual = self.mesh.visual
        if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
            # Use per-vertex colors, repeat for face corners
            flat_colors = visual.vertex_colors[faces.flatten()].astype(np.float32)
            if flat_colors.max() > 1.0:
                flat_colors /= 255.0
        else:
            # Default white
            flat_colors = np.ones((len(flat_vertices), 4), dtype=np.float32)
        
        # UVs from visual
        if hasattr(visual, "uv") and visual.uv is not None:
            uv = visual.uv
            # UV layout: per-vertex or per-face-corner
            if len(uv) == len(self.mesh.vertices):
                flat_uvs = uv[faces.flatten()]
            elif len(uv) == len(faces) * 3:
                flat_uvs = uv
            else:
                flat_uvs = np.zeros((len(flat_vertices), 2), dtype=np.float32)
        else:
            # Fallback to _uv for compatibility
            if hasattr(self, "_uv") and self._uv is not None:
                uv = self._uv
                if len(uv) == len(self.mesh.vertices):
                    flat_uvs = uv[faces.flatten()]
                elif len(uv) == len(faces) * 3:
                    flat_uvs = uv
                else:
                    flat_uvs = np.zeros((len(flat_vertices), 2), dtype=np.float32)
            else:
                flat_uvs = np.zeros((len(flat_vertices), 2), dtype=np.float32)
            
        return flat_vertices, flat_normals, flat_colors, flat_uvs

    # =========================================================================
    # GPU methods (called by renderer)
    # =========================================================================
    
    def _init_gpu(self, ctx: 'moderngl.Context', program: 'moderngl.Program'):
        """Initialize GPU resources. Called by renderer."""
        flat_vertices, flat_normals, flat_colors, flat_uvs = self._get_flattened_geometry()
        
        if flat_vertices is None:
             raise RuntimeError("Object has no geometry loaded")
        
        # Interleave data
        vertex_data = np.hstack([flat_vertices, flat_normals, flat_colors, flat_uvs]).astype(np.float32)
        
        # Create GPU buffers
        self._vbo = ctx.buffer(vertex_data.tobytes())
        self._vao = ctx.vertex_array(
            program,
            [(self._vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')]
        )
        self._gpu_initialized = True
    
    def _release_gpu(self):
        """Release GPU resources."""
        if self._vao:
            self._vao.release()
            self._vao = None
        if self._vbo:
            self._vbo.release()
            self._vbo = None
        self._gpu_initialized = False

    def _rotation_matrix(self):
        cx, cy, cz = np.cos(self._rotation)
        sx, sy, sz = np.sin(self._rotation)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        return Rx @ Ry @ Rz

    def __repr__(self):
        return f"Object3D(name='{hash(self)}', position={self.position}, scale={self.scale})"


# =============================================================================
# Primitive factory functions
# =============================================================================

def create_cube(size: float = 1.0, 
                position: Tuple[float, float, float] = (0, 0, 0),
                color: Optional[ColorType] = None) -> Object3D:
    """
    Create a cube primitive (add Collider separately if needed).
    
    Args:
        size: Cube size
        position: Cube position
        color: Cube color
    
    Returns:
        Object3D cube
    """
    obj = Object3D(position=position, color=color)
    
    s = size / 2
    vertices = np.array([
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        [-s, -s, -s], [-s,  s, -s], [ s,  s, -s], [ s, -s, -s],
        [-s,  s, -s], [-s,  s,  s], [ s,  s,  s], [ s,  s, -s],
        [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],
        [ s, -s, -s], [ s,  s, -s], [ s,  s,  s], [ s, -s,  s],
        [-s, -s, -s], [-s, -s,  s], [-s,  s,  s], [-s,  s, -s],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.int32)
    
    # Use trimesh object (default white vertex colors in renderer for tinting)
    obj.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    obj._post_process_geometry(f"primitive_cube_{size}")
    
    return obj


def create_sphere(radius: float = 1.0, 
                  subdivisions: int = 2,
                  position: Tuple[float, float, float] = (0, 0, 0),
                  color: Optional[ColorType] = None) -> Object3D:
    """
    Create a sphere primitive.
    """
    obj = Object3D(position=position, color=color)
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    obj.mesh = mesh
    obj._post_process_geometry(f"primitive_sphere_{radius}")
    return obj


def create_plane(width: float = 10.0, 
                 height: float = 10.0,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 color: Optional[ColorType] = None) -> Object3D:
    """
    Create a horizontal plane primitive (add Collider separately if needed).
    
    Args:
        width: Plane width (X axis)
        height: Plane height (Z axis)
        position: Plane position
        color: Plane color
    
    Returns:
        Object3D plane
    """
    obj = Object3D(position=position, color=color)
    
    w, h = width / 2, height / 2
    vertices = np.array([
        [-w, 0, -h],
        [ w, 0, -h],
        [ w, 0,  h],
        [-w, 0,  h],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
    ], dtype=np.int32)

    # Use trimesh object (default white vertex colors in renderer for tinting)
    obj.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    obj._post_process_geometry(f"primitive_plane_{width}_{height}")

    return obj
 