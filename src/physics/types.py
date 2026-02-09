from enum import IntEnum

class ColliderType(IntEnum):
    SPHERE = 0
    CYLINDER = 1
    CUBE = 2
    MESH = 3

    @staticmethod
    def all():
        return [
            ColliderType.SPHERE,
            ColliderType.CYLINDER,
            ColliderType.CUBE,
            ColliderType.MESH
        ]


class CollisionRelation(IntEnum):
    """Collision relation (IGNORE/TRIGGER/SOLID) between ColliderGroups."""
    IGNORE = 0
    TRIGGER = 1
    SOLID = 2


class CollisionMode(IntEnum):
    # Per-collider mode:
    # IGNORE: no detection
    # TRIGGER: detect but pass through (no block)
    # NORMAL: detect + block (solid)
    # CONTINUOUS: detect + block with sweep
    NORMAL = 0
    CONTINUOUS = 1
    IGNORE = 2
    TRIGGER = 3

# Layer/mask system (Unity-style for filtering; 32 layers)
# Collider.layer: int (0-31, default 0)
# Collider.collision_mask: int (bitmask, default all layers collide)
# If (layer & other.collision_mask) == 0 or vice-versa: ignore (no detect/block)
