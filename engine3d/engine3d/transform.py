from typing import Optional, TYPE_CHECKING, List
import numpy as np

from engine3d.engine3d.component import Component
from engine3d.types import Vector3, Vector3Like

if TYPE_CHECKING:
    from .gameobject import GameObject


class Transform(Component):
    """Component storing position, rotation, and scale.
    
    Supports parent-child relationships where children's transforms are relative to parent.
    World position, rotation, and scale are computed from parent + local values.
    """

    def __init__(self):
        super().__init__()
        # Local transform values (relative to parent)
        self._local_position = Vector3.zero()
        self._local_rotation = np.zeros(3, dtype=np.float32)
        self._local_scale = Vector3.one()
        
        # Cached world transform values
        self._world_position = Vector3.zero()
        self._world_rotation = np.zeros(3, dtype=np.float32)
        self._world_scale = Vector3.one()

        self._transform_dirty = True
        self._world_dirty = True
        self._cached_model = None
        self._cached_rotation = None
        self._prev_position = Vector3(self._local_position)
        
        # Parent-child relationships
        self._parent: Optional['Transform'] = None
        self._children: List['Transform'] = []

    def _mark_dirty(self):
        self._transform_dirty = True
        self._world_dirty = True
        # Mark all children as dirty too
        for child in self._children:
            child._mark_dirty()
        if self.game_object:
            from engine3d.physics import Collider
            for comp in self.game_object.get_components(Collider):
                comp._transform_dirty = True

    def _update_prev_position(self):
        self._prev_position = Vector3(self._local_position)
    
    # =========================================================================
    # Parent-child relationship methods
    # =========================================================================
    
    @property
    def parent(self) -> Optional['Transform']:
        """Get the parent transform."""
        return self._parent
    
    @parent.setter
    def parent(self, value: Optional['Transform']):
        """Set the parent transform."""
        if self._parent is value:
            return
        
        # Remove from old parent
        if self._parent is not None:
            self._parent._children.remove(self)
        
        self._parent = value
        
        # Add to new parent
        if self._parent is not None:
            self._parent._children.append(self)
        
        self._mark_dirty()
    
    @property
    def children(self) -> List['Transform']:
        """Get list of child transforms."""
        return self._children.copy()
    
    def add_child(self, child: 'Transform'):
        """Add a child transform."""
        child.parent = self
    
    def remove_child(self, child: 'Transform'):
        """Remove a child transform."""
        if child in self._children:
            child.parent = None
    
    def detach_from_parent(self):
        """Detach this transform from its parent."""
        self.parent = None
    
    # =========================================================================
    # Local transform properties (relative to parent)
    # =========================================================================
    
    @property
    def local_position(self) -> Vector3:
        """Get local position (relative to parent)."""
        return Vector3(self._local_position)
    
    @local_position.setter
    def local_position(self, value):
        self._update_prev_position()
        self._local_position = Vector3(value)
        self._mark_dirty()
    
    @property
    def local_rotation(self) -> tuple:
        """Get local rotation in degrees (relative to parent)."""
        return tuple(np.degrees(self._local_rotation))
    
    @local_rotation.setter
    def local_rotation(self, value):
        self._local_rotation = np.radians(value).astype(np.float32)
        self._mark_dirty()
    
    @property
    def local_scale(self) -> Vector3:
        """Get local scale (relative to parent)."""
        return Vector3(self._local_scale)
    
    @local_scale.setter
    def local_scale(self, value):
        self._local_scale = Vector3(value)
        self._mark_dirty()
    
    # =========================================================================
    # World transform properties (computed from parent + local)
    # =========================================================================
    
    def _compute_world_transform(self):
        """Compute world position, rotation, and scale from parent."""
        if not self._world_dirty:
            return
        if self._parent is None:
            self._world_position = Vector3(self._local_position)
            self._world_rotation = self._local_rotation.copy()
            self._world_scale = Vector3(self._local_scale)
        else:
            # Get parent's world transform
            parent = self._parent
            parent._compute_world_transform()
            
            # World scale = parent scale * local scale
            self._world_scale = Vector3.scale(parent._world_scale, self._local_scale)
            
            # World rotation = parent rotation + local rotation
            self._world_rotation = parent._world_rotation + self._local_rotation
            
            # World position = parent position + (parent rotation applied to local position * parent scale)
            # First, rotate local position by parent's rotation
            cx, cy, cz = np.cos(parent._world_rotation)
            sx, sy, sz = np.sin(parent._world_rotation)
            
            # Build rotation matrix from Euler angles (XYZ order - intrinsic rotations)
            # For intrinsic rotations, we apply rotations in XYZ order: Rx @ Ry @ Rz
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            R = Rx @ Ry @ Rz  # XYZ intrinsic rotation order
            
            # Scale local position by parent scale, then rotate, then translate
            scaled_local = self._local_position.to_numpy() * parent._world_scale.to_numpy()
            rotated_local = scaled_local @ R
            self._world_position = parent._world_position + rotated_local
        self._world_dirty = False
    
    @property
    def world_position(self) -> Vector3:
        """Get world position (computed from parent + local)."""
        self._compute_world_transform()
        return Vector3(self._world_position)
    
    @world_position.setter
    def world_position(self, value):
        """Set world position (converts to local based on parent)."""
        world_pos = Vector3(value)
        
        if self._parent is None:
            self._local_position = world_pos
        else:
            # Convert world position to local position
            parent = self._parent
            parent._compute_world_transform()
            
            # Reverse the transformation: local = inv(rotate) * (world - parent) / scale
            # Inverse of Rx @ Ry @ Rz is Rz^T @ Ry^T @ Rx^T = Rz(-z) @ Ry(-y) @ Rx(-x)
            cx, cy, cz = np.cos(-parent._world_rotation)
            sx, sy, sz = np.sin(-parent._world_rotation)
            
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            R_inv = Rz @ Ry @ Rx  # Inverse of XYZ is ZYX with negated angles
            
            delta = (world_pos - parent._world_position).to_numpy()
            rotated_delta = delta @ R_inv
            self._local_position = Vector3(rotated_delta / parent._world_scale.to_numpy())
        
        self._mark_dirty()
    
    @property
    def world_rotation(self) -> tuple:
        """Get world rotation in degrees (computed from parent + local)."""
        self._compute_world_transform()
        return tuple(np.degrees(self._world_rotation))
    
    @world_rotation.setter
    def world_rotation(self, value):
        """Set world rotation (converts to local based on parent)."""
        world_rot = np.radians(value).astype(np.float32)
        
        if self._parent is None:
            self._local_rotation = world_rot
        else:
            parent = self._parent
            parent._compute_world_transform()
            self._local_rotation = world_rot - parent._world_rotation
        
        self._mark_dirty()
    
    @property
    def world_scale(self) -> Vector3:
        """Get world scale (computed from parent + local)."""
        self._compute_world_transform()
        return Vector3(self._world_scale)

    @world_scale.setter
    def world_scale(self, value):
        """Set world scale (converts to local based on parent)."""
        world_scale = Vector3(value)

        if self._parent is None:
            self._local_scale = world_scale
        else:
            parent = self._parent
            parent._compute_world_transform()
            self._local_scale = Vector3(world_scale.to_numpy() / parent._world_scale.to_numpy())

        self._mark_dirty()
    
    # =========================================================================
    # Convenience properties (alias to local transform for backward compatibility)
    # =========================================================================
    
    @property
    def position(self) -> Vector3:
        """Get local position (alias for local_position)."""
        return Vector3(self._local_position)

    @position.setter
    def position(self, value):
        self._update_prev_position()
        self._local_position = Vector3(value)
        self._mark_dirty()

    @property
    def x(self) -> float:
        return float(self._local_position.x)

    @x.setter
    def x(self, value: float):
        self.position = (value, self._local_position.y, self._local_position.z)

    @property
    def y(self) -> float:
        return float(self._local_position.y)

    @y.setter
    def y(self, value: float):
        self.position = (self._local_position.x, value, self._local_position.z)

    @property
    def z(self) -> float:
        return float(self._local_position.z)

    @z.setter
    def z(self, value: float):
        self.position = (self._local_position.x, self._local_position.y, value)

    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        self._update_prev_position()
        self._local_position = self._local_position + Vector3(dx, dy, dz)
        self._mark_dirty()

    @property
    def rotation(self) -> tuple:
        return tuple(np.degrees(self._local_rotation))

    @rotation.setter
    def rotation(self, value):
        self._local_rotation = np.radians(value).astype(np.float32)
        self._mark_dirty()

    @property
    def rotation_x(self) -> float:
        return float(np.degrees(self._local_rotation[0]))

    @rotation_x.setter
    def rotation_x(self, value: float):
        self._local_rotation[0] = np.radians(value)
        self._mark_dirty()

    @property
    def rotation_y(self) -> float:
        return float(np.degrees(self._local_rotation[1]))

    @rotation_y.setter
    def rotation_y(self, value: float):
        self._local_rotation[1] = np.radians(value)
        self._mark_dirty()

    @property
    def rotation_z(self) -> float:
        return float(np.degrees(self._local_rotation[2]))

    @rotation_z.setter
    def rotation_z(self, value: float):
        self._local_rotation[2] = np.radians(value)
        self._mark_dirty()

    def rotate(self, dx: float = 0, dy: float = 0, dz: float = 0):
        self._local_rotation += np.radians([dx, dy, dz]).astype(np.float32)
        self._mark_dirty()

    @property
    def scale(self) -> float:
        return float(self._local_scale.x)

    @scale.setter
    def scale(self, value: float):
        self._local_scale = Vector3(value, value, value)
        self._mark_dirty()

    @property
    def scale_xyz(self) -> Vector3:
        return Vector3(self._local_scale)

    @scale_xyz.setter
    def scale_xyz(self, value):
        self._local_scale = Vector3(value)
        self._mark_dirty()

    def get_model_matrix(self) -> np.ndarray:
        if not self._transform_dirty:
            return self._cached_model
        
        # Compute world transform first
        self._compute_world_transform()

        # Get rotation matrix (cached in _compute_world_transform or computed here if needed)
        # Recomputing R here to be safe and consistent with previous code
        cx, cy, cz = np.cos(self._world_rotation)
        sx, sy, sz = np.sin(self._world_rotation)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rx @ Ry @ Rz
        self._cached_rotation = R

        s_x, s_y, s_z = self._world_scale.to_tuple()
        tx, ty, tz = self._world_position.to_tuple()
        S = np.array([[s_x, 0, 0, 0], [0, s_y, 0, 0], [0, 0, s_z, 0], [0, 0, 0, 1]], dtype=np.float32)
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = R
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [tx, ty, tz, 1]], dtype=np.float32)
        self._cached_model = S @ R4 @ T
        self._transform_dirty = False
        return self._cached_model

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation matrix (world space)."""
        # Ensure cached rotation is up to date
        self.get_model_matrix() 
        return self._cached_rotation

    @property
    def forward(self) -> np.ndarray:
        """Get forward vector (world space, assuming -Z is forward)."""
        return -self.rotation_matrix[2, :]

    @property
    def backward(self) -> np.ndarray:
        """Get backward vector (world space, +Z)."""
        return self.rotation_matrix[2, :]

    @property
    def right(self) -> np.ndarray:
        """Get right vector (world space, +X)."""
        return self.rotation_matrix[0, :]

    @property
    def left(self) -> np.ndarray:
        """Get left vector (world space, -X)."""
        return -self.rotation_matrix[0, :]

    @property
    def up(self) -> np.ndarray:
        """Get up vector (world space, +Y)."""
        return self.rotation_matrix[1, :]

    @property
    def down(self) -> np.ndarray:
        """Get down vector (world space, -Y)."""
        return -self.rotation_matrix[1, :]

    def look_at(self, target: 'Vector3Like', world_up: 'Vector3Like' = (0, 1, 0)):
        """Look at a target position."""
        eye = self.world_position
        target = Vector3(target)
        world_up = Vector3(world_up)
        
        # Forward vector (from eye to target)
        # Note: Camera looks down -Z, so forward is target - eye
        f = target - eye
        dist = f.magnitude
        if dist < 1e-6:
            return
        f = f.normalized
        
        # Right vector
        r = Vector3.cross(f, world_up)
        if r.magnitude < 1e-6:
            # Handle case where looking straight up/down
            r = Vector3.right()
        else:
            r = r.normalized
            
        # Up vector
        u = Vector3.cross(r, f)
        
        # Create rotation matrix [r, u, -f]
        # This transforms local basis to world basis
        R = np.vstack([r.to_numpy(), u.to_numpy(), (-f).to_numpy()])
        
        # Extract Euler angles from rotation matrix
        # Assuming XYZ order (Rx @ Ry @ Rz)
        # R = [ [cy*cz,              -cy*sz,               sy    ],
        #       [cx*sz + sx*sy*cz,   cx*cz - sx*sy*sz,    -sx*cy ],
        #       [sx*sz - cx*sy*cz,   sx*cz + cx*sy*sz,     cx*cy ] ]
        # Wait, my rotation matrix construction in _compute_world_transform might be different.
        # Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
        # Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
        # Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
        # R = Rx @ Ry @ Rz
        
        # Let's use scipy or a robust method if available, or just simple extraction
        # Sy = R[0, 2]
        # cy = sqrt(1 - sy^2)
        # if cy > 1e-6:
        #     sx = -R[1, 2] / cy
        #     cx = R[2, 2] / cy
        #     sz = -R[0, 1] / cy
        #     cz = R[0, 0] / cy
        # else:
        #     # Gimbal lock
        #     ...
        
        # Simplified extraction for YXZ order which is common? No, I used XYZ.
        # Let's reverse engineer my R construction:
        # R = Rx @ Ry @ Rz
        # = [ [cy*cz,              -cy*sz,               sy    ],
        #     [cx*sz + sx*sy*cz,   cx*cz - sx*sy*sz,    -sx*cy ],
        #     [sx*sz - cx*sy*cz,   sx*cz + cx*sy*sz,     cx*cy ] ]
        # Note: This matches standard XYZ intrinsic if row/col vectors are correct.
        
        # sy = R[0, 2]
        sy = R[0, 2]
        if sy < 1.0:
            if sy > -1.0:
                cy = np.sqrt(1 - sy*sy)
                rotation_y = np.arcsin(sy)
                rotation_x = np.arctan2(-R[1, 2], R[2, 2])
                rotation_z = np.arctan2(-R[0, 1], R[0, 0])
            else:
                # sy = -1
                rotation_y = -np.pi / 2
                rotation_x = -np.arctan2(R[1, 0], R[1, 1])
                rotation_z = 0
        else:
            # sy = 1
            rotation_y = np.pi / 2
            rotation_x = np.arctan2(R[1, 0], R[1, 1])
            rotation_z = 0
            
        self.world_rotation = (np.degrees(rotation_x), np.degrees(rotation_y), np.degrees(rotation_z))
