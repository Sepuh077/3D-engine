import numpy as np
from typing import TYPE_CHECKING
from .types import CollisionMode, CollisionRelation
from src.engine3d.component import Component
from .group import ColliderGroup

if TYPE_CHECKING:
    from src.engine3d.gameobject import GameObject


class Collider(Component):
    """Base collider. Subclasses for types (Box, Sphere, Capsule). Contains Object3D ref."""

    def __init__(self):
        super().__init__()
        self.center = [0.0, 0.0, 0.0]
        self.sphere = None
        self.obb = None
        self.aabb = None
        self.cylinder = None
        self.mesh_data = None
        self.collision_mode = CollisionMode.NORMAL
        # Group for relations (default for all colliders)
        self.group = ColliderGroup._registry.get("default") or ColliderGroup("default")
        # Per-collider collisions tracking
        self._current_collisions: set = set()
        # Dirty flag (shared transform dirty from Object3D)
        self._transform_dirty = True

    def set_bounds_data(self, sphere, obb, aabb, cylinder, mesh_data=None):
        self.sphere = sphere
        self.obb = obb
        self.aabb = aabb
        self.cylinder = cylinder
        self.mesh_data = mesh_data

    # Shared compute (main part used by all subs; called by their update_bounds)
    # Subs override for their specific (only needed calc/params; no unwanted e.g. radius for Box)
    def _compute_shared(self):
        if not self._transform_dirty or not self.game_object:
            return None
        obj = self.game_object
        from src.engine3d.object3d import Object3D
        obj3d = obj.get_component(Object3D)
        if not obj3d or obj3d.mesh is None:
            self._transform_dirty = False
            return None

        obj.transform._compute_world_transform()

        # Shared rotation/extents/center
        rotation = obj.transform._world_rotation
        scale = obj.transform._world_scale
        position = obj.transform._world_position

        cx, cy, cz = np.cos(rotation)
        sx, sy, sz = np.sin(rotation)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rx @ Ry @ Rz

        local_extents = (obj3d._local_max - obj3d._local_min) * 0.5
        extents = local_extents * scale
        local_center = (obj3d._local_min + obj3d._local_max) * 0.5
        center_offset = (local_center * scale) @ R
        base_center = position + center_offset
        absR = np.abs(R)
        half_extents = absR @ extents
        aabb_dims = half_extents * 2

        # Collider-specific center offset (local offset scaled/rotated)
        local_offset = local_extents * np.array(self.center, dtype=np.float32)
        c_offset = (local_offset * scale) @ R
        collider_center = base_center + c_offset

        # Keep collision bounds in sync when no custom center set
        if np.allclose(local_offset, 0.0):
            collider_center = base_center

        # Mesh data if needed
        if obj3d.mesh is not None:
            model = obj.transform.get_model_matrix()
            mesh_data = (obj3d.mesh.vertices, obj3d.mesh.faces, model)
            self.mesh_data = mesh_data

        self._transform_dirty = False
        return R, absR, extents, aabb_dims, collider_center

    def update_bounds(self):
        # Base only shared; subs override to add their specific (only needed)
        shared = self._compute_shared()
        if shared is None:
            return
        # (subs extend here)
        pass

    def get_world_sphere(self):
        self.update_bounds()
        return self.sphere

    def get_world_obb(self):
        self.update_bounds()
        return self.obb
    
    def get_world_aabb(self):
        self.update_bounds()
        return self.aabb

    def get_world_cylinder(self):
        self.update_bounds()
        return self.cylinder
    
    def get_mesh_data(self):
        self.update_bounds()
        return self.mesh_data

    # Collision helpers (moved here; collider-centric)
    def check_collision(self, other: 'Collider') -> bool:
        if other is None or not self.game_object or not other.game_object:
            return False
        # Use ColliderGroup: IGNORE skips (Trigger=detect/pass, Normal=block)
        if self.group.get_relation(other.group) == CollisionRelation.IGNORE:
            return False
        self.update_bounds()
        other.update_bounds()
        from src.physics.collision import objects_collide
        return objects_collide(self, other)

    def contains_point(self, point, radius=1.0):
        if not self.game_object:
            return False
        self.update_bounds()
        from src.physics.collision import collide_point_with_radius
        return collide_point_with_radius(np.array(point, dtype=np.float32), self, radius)

    def OnCollisionEnter(self, other):
        pass

    def OnCollisionExit(self, other):
        pass

    def OnCollisionStay(self, other):
        pass


class BoxCollider(Collider):
    """Box/OBB collider (replaces old CUBE). Only size/center."""

    def __init__(self, center=None, size=None):
        super().__init__()
        if center:
            self.center = center
        self.size = size or [1.0, 1.0, 1.0]
        self.type = 2  # legacy for compat in collision funcs

    # Override: only Box/OBB (no radius/cylinder)
    def update_bounds(self):
        shared = self._compute_shared()
        if shared is None:
            return
        R, absR, extents, aabb_dims, collider_center = shared
        # Box-specific
        obb_extents = extents * np.array(self.size, dtype=np.float32)
        obb = (collider_center, R, obb_extents)
        half = absR @ obb_extents
        aabb = (collider_center - half, collider_center + half)
        self.obb = obb
        self.aabb = aabb
        # (no sphere/cylinder)


class SphereCollider(Collider):
    """Sphere collider. Only radius/center."""

    def __init__(self, center=None, radius=1.0):
        super().__init__()
        if center:
            self.center = center
        self.radius = radius
        self.type = 0  # legacy

    # Override: only Sphere (no size/height)
    def update_bounds(self):
        shared = self._compute_shared()
        if shared is None:
            return
        R, absR, extents, aabb_dims, collider_center = shared
        # Sphere-specific
        obj = self.game_object
        from src.engine3d.object3d import Object3D
        obj3d = obj.get_component(Object3D)
        radius = obj3d._local_radius * np.max(np.abs(obj.transform._world_scale)) * self.radius
        sphere = (collider_center, float(radius))
        # AABB from sphere approx
        aabb = (collider_center - radius, collider_center + radius)
        self.sphere = sphere
        self.aabb = aabb
        # (no obb/cylinder)


class CapsuleCollider(Collider):
    """Capsule/cylinder collider. Only radius/height/center."""

    def __init__(self, center=None, radius=1.0, height=1.0):
        super().__init__()
        if center:
            self.center = center
        self.radius = radius
        self.height = height
        self.type = 1  # legacy

    # Override: only Cylinder (no size)
    def update_bounds(self):
        shared = self._compute_shared()
        if shared is None:
            return
        R, absR, extents, aabb_dims, collider_center = shared
        # Cylinder-specific
        obj = self.game_object
        from src.engine3d.object3d import Object3D
        obj3d = obj.get_component(Object3D)
        half_ext = (obj3d._local_max - obj3d._local_min) * 0.5 * np.abs(obj.transform._world_scale)
        cyl_radius = float(np.maximum(half_ext[0], half_ext[2])) * self.radius
        half_height = float(half_ext[1]) * self.height
        cylinder = (collider_center, cyl_radius, half_height)
        # AABB approx
        aabb = (collider_center - np.array([cyl_radius, half_height, cyl_radius]), collider_center + np.array([cyl_radius, half_height, cyl_radius]))
        self.cylinder = cylinder
        self.aabb = aabb
        # (no sphere/obb)
