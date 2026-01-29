"""
Object3D - 3D object that can be loaded, positioned, rotated, and scaled.
"""
import hashlib
import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING

from src.physics import ColliderType
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
        collider_type: str = ColliderType.CUBE,
    ):
        # ---------------- Transform ----------------
        self._position = np.array(position, dtype=np.float32)
        self._rotation = np.zeros(3, dtype=np.float32)  # radians
        self._scale = np.array([scale, scale, scale], dtype=np.float32)

        self._transform_dirty = True

        # Cached transforms / colliders
        self._cached_rotation = None
        self._cached_model = None
        self._cached_aabb = None
        self._cached_obb = None
        self._cached_sphere = None
        self._cached_cylinder = None

        # ---------------- Geometry ----------------
        self._vertices = None
        self._faces = None
        self._normals = None

        self._local_min = None
        self._local_max = None
        self._local_radius = None

        # ---------------- Misc ----------------
        self._color = np.array(color if color else (1, 1, 1), dtype=np.float32)
        self._visible = True
        self._collider_type = collider_type
        self._static = False
        self.draw_bounding_box = False

        # GPU handles (initialized later)
        self._vbo = None
        self._vao = None
        self._gpu_initialized = False

        # Mesh identity for batching/instancing
        self._mesh_key = None
        self._mesh = None

        if filename:
            self.load(filename)

    # ======================================================================
    # Loading & geometry preprocessing (ONCE)
    # ======================================================================

    def load(self, filename: str):
        vertices, faces = [], []

        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    vertices.append(list(map(float, line.split()[1:4])))
                elif line.startswith("f "):
                    idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])

        self._vertices = np.array(vertices, dtype=np.float32)
        self._faces = np.array(faces, dtype=np.int32)

        center = self._vertices.mean(axis=0)
        self._vertices -= center

        self._local_min = self._vertices.min(axis=0)
        self._local_max = self._vertices.max(axis=0)
        self._local_radius = np.linalg.norm(self._vertices, axis=1).max()

        self._compute_normals()

        self._mesh_key = ("obj", filename)
        self._transform_dirty = True
    
    def _compute_normals(self):
        normals = np.zeros_like(self._vertices)
        for face in self._faces:
            v0, v1, v2 = self._vertices[face]
            n = np.cross(v1 - v0, v2 - v0)
            n /= max(np.linalg.norm(n), 1e-6)
            normals[face] += n
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        self._normals = normals / np.maximum(norms, 1e-6)

    # ======================================================================
    # Dirty flag helpers
    # ======================================================================

    def _mark_dirty(self):
        self._transform_dirty = True

    # =========================================================================
    # Position properties
    # =========================================================================
    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

        self._local_min = self._vertices.min(axis=0)
        self._local_max = self._vertices.max(axis=0)

        # Bounding sphere in local space
        self._local_center = np.zeros(3, dtype=np.float32)
        self._local_radius = np.linalg.norm(self._vertices, axis=1).max()
        self._mesh_key = None
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Get position as tuple."""
        return tuple(self._position)
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set position."""
        self._position = np.array(value, dtype=np.float32)
        self._mark_dirty()
    
    @property
    def x(self) -> float:
        return float(self._position[0])
    
    @x.setter
    def x(self, value: float):
        self._position[0] = value
        self._mark_dirty()
    
    @property
    def y(self) -> float:
        return float(self._position[1])
    
    @y.setter
    def y(self, value: float):
        self._position[1] = value
        self._mark_dirty()
    
    @property
    def z(self) -> float:
        return float(self._position[2])
    
    @z.setter
    def z(self, value: float):
        self._position[2] = value
        self._mark_dirty()
    
    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move object by offset."""
        self._position += np.array([dx, dy, dz], dtype=np.float32)
        self._mark_dirty()
    
    # =========================================================================
    # Rotation properties (in degrees for user convenience)
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
    # Collider properties
    # =========================================================================

    @property
    def collider_type(self) -> str:
        """Type of collider: cube (OBB), sphere or cylinder."""
        return self._collider_type

    @collider_type.setter
    def collider_type(self, value: ColliderType):
        self._collider_type = value
    
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
        self._color = np.array(value, dtype=np.float32)
    
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

    # =========================================================================
    # Mesh identity (for caching/instancing)
    # =========================================================================

    def get_mesh_key(self):
        """
        Return a stable mesh key for batching/instancing.
        Falls back to hashing geometry if no explicit key is set.
        """
        if self._mesh_key is not None:
            return self._mesh_key

        if self._vertices is None or self._faces is None:
            return None

        h = hashlib.blake2b(digest_size=16)
        h.update(self._vertices.tobytes())
        h.update(self._faces.tobytes())
        self._mesh_key = ("geom", h.hexdigest())
        return self._mesh_key
    
    # =========================================================================
    # Model matrix
    # =========================================================================
    
    def get_model_matrix(self) -> np.ndarray:
        self._update_cache()
        return self._cached_model
    
    # =========================================================================
    # GPU methods (called by renderer)
    # =========================================================================
    
    def _init_gpu(self, ctx: 'moderngl.Context', program: 'moderngl.Program'):
        """Initialize GPU resources. Called by renderer."""
        if self._vertices is None:
            raise RuntimeError("Object has no geometry loaded")
        
        # Flatten vertices for rendering
        flat_vertices = self._vertices[self._faces.flatten()]
        flat_normals = self._normals[self._faces.flatten()]
        
        # Interleave data
        vertex_data = np.hstack([flat_vertices, flat_normals]).astype(np.float32)
        
        # Create GPU buffers
        self._vbo = ctx.buffer(vertex_data.tobytes())
        self._vao = ctx.vertex_array(
            program,
            [(self._vbo, '3f 3f', 'in_position', 'in_normal')]
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

    def _update_cache(self):
        if not self._transform_dirty:
            return

        # ----- Rotation matrix (ONCE) -----
        cx, cy, cz = np.cos(self._rotation)
        sx, sy, sz = np.sin(self._rotation)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

        R = Rx @ Ry @ Rz
        self._cached_rotation = R

        extents = (self._local_max - self._local_min) * 0.5 * self._scale
        local_center = (self._local_min + self._local_max) * 0.5

        center_offset = R @ (local_center * self._scale)
        center = self._position + center_offset

        # ----- OBB -----
        self._cached_obb = (center, R, extents)

        # ----- Sphere -----
        radius = self._local_radius * np.max(np.abs(self._scale))
        self._cached_sphere = (center, float(radius))

        # ----- AABB from OBB (fast way) -----
        absR = np.abs(R)
        half = absR @ extents
        self._cached_aabb = (center - half, center + half)

        # ----- Cylinder -----
        half_ext = (self._local_max - self._local_min) * 0.5 * np.abs(self._scale)

        cyl_radius = float(np.maximum(half_ext[0], half_ext[2]))
        half_height = float(half_ext[1])

        self._cached_cylinder = (center, cyl_radius, half_height)

        # ----- Model matrix (row-major, matches view/projection math) -----
        sx, sy, sz = self._scale
        tx, ty, tz = self._position

        S = np.array([
            [sx, 0,  0,  0],
            [0,  sy, 0,  0],
            [0,  0,  sz, 0],
            [0,  0,  0,  1],
        ], dtype=np.float32)

        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = R

        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [tx, ty, tz, 1],
        ], dtype=np.float32)

        model = S @ R4 @ T

        self._cached_model = model

        self._transform_dirty = False


    def _rotation_matrix(self):
        cx, cy, cz = np.cos(self._rotation)
        sx, sy, sz = np.sin(self._rotation)

        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], dtype=np.float32)

        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype=np.float32)

        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        return Rx @ Ry @ Rz

    def world_aabb(self):
        self._update_cache()
        return self._cached_aabb

    def world_obb(self):
        self._update_cache()
        return self._cached_obb

    def world_sphere(self):
        self._update_cache()
        return self._cached_sphere

    def world_cylinder(self):
        self._update_cache()
        return self._cached_cylinder

    # def get_model_matrix(self):
    #     self._update_cache()
    #     return self._cached_model

    def draw_collider(self, window: 'Window3D', color: Tuple[float, float, float] = (0, 1, 0), line_width: float = 1.0):
        """
        Convenience to draw this object's collider via a Window3D.
        """
        window.draw_collider(self, color=color, line_width=line_width)

    # =========================================================================
    # Collision helpers
    # =========================================================================

    def check_collision(self, other: 'Object3D') -> bool:
        """
        Check collision with another object based on collider types.
        """
        if other is None:
            return False
        from src.physics.collision import objects_collide
        return objects_collide(self, other)

    
    def __repr__(self):
        return f"Object3D(name='{hash(self)}', position={self.position}, scale={self.scale})"


# =============================================================================
# Primitive factory functions
# =============================================================================

def create_cube(size: float = 1.0, 
                position: Tuple[float, float, float] = (0, 0, 0),
                color: Optional[ColorType] = None,
                collider_type: str = ColliderType.CUBE) -> Object3D:
    """
    Create a cube primitive.
    
    Args:
        size: Cube size
        position: Cube position
        color: Cube color
    
    Returns:
        Object3D cube
    """
    obj = Object3D(position=position, color=color, collider_type=collider_type)
    
    s = size / 2
    vertices = np.array([
        # Front face
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        # Back face
        [-s, -s, -s], [-s,  s, -s], [ s,  s, -s], [ s, -s, -s],
        # Top face
        [-s,  s, -s], [-s,  s,  s], [ s,  s,  s], [ s,  s, -s],
        # Bottom face
        [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],
        # Right face
        [ s, -s, -s], [ s,  s, -s], [ s,  s,  s], [ s, -s,  s],
        # Left face
        [-s, -s, -s], [-s, -s,  s], [-s,  s,  s], [-s,  s, -s],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],       # Front
        [4, 5, 6], [4, 6, 7],       # Back
        [8, 9, 10], [8, 10, 11],    # Top
        [12, 13, 14], [12, 14, 15], # Bottom
        [16, 17, 18], [16, 18, 19], # Right
        [20, 21, 22], [20, 22, 23], # Left
    ], dtype=np.int32)
    
    obj.vertices = vertices
    obj._faces = faces
    obj._center = np.zeros(3, dtype=np.float32)
    obj._compute_normals()
    obj._mesh_key = ("primitive", ColliderType.CUBE, float(size))
    
    return obj


def create_plane(width: float = 10.0, 
                 height: float = 10.0,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 color: Optional[ColorType] = None,
                 collider_type: str = ColliderType.CUBE) -> Object3D:
    """
    Create a horizontal plane primitive.
    
    Args:
        width: Plane width (X axis)
        height: Plane height (Z axis)
        position: Plane position
        color: Plane color
    
    Returns:
        Object3D plane
    """
    obj = Object3D(position=position, color=color, collider_type=collider_type)
    
    w, h = width / 2, height / 2
    vertices = np.array([
        [-w, 0, -h],
        [ w, 0, -h],
        [ w, 0,  h],
        [-w, 0,  h],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 2, 1],  # Facing up
        [0, 3, 2],
    ], dtype=np.int32)
    
    obj.vertices = vertices
    obj._faces = faces
    obj._center = np.zeros(3, dtype=np.float32)
    obj._compute_normals()
    
    return obj
