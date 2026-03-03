"""
Particle system implementation inspired by Unity-style emitters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import random

import numpy as np

from src.physics import BoxCollider, SphereCollider, CollisionMode, Collider
from .color import ColorType
from .object3d import Object3D, create_cube


ParticleObject = Union[str, Object3D, Callable[[], Object3D]]
LifetimeFloat = Callable[[float], float]
LifetimeColor = Callable[[float], ColorType]
LifetimeVelocity = Callable[[float], Union[float, np.ndarray, tuple, list]]


@dataclass
class ParticleBurst:
    """Burst emission configuration."""

    interval: float = 1.0
    count: int = 10
    randomize: bool = False


class Particle:
    """Internal particle instance."""

    def __init__(self, obj: Object3D):
        self.obj = obj
        self.velocity = np.zeros(3, dtype=np.float32)
        self.life = 1.0
        self.age = 0.0
        self.active = False


class ParticleShape:
    """Base class for emission shapes."""

    def get_spawn_pos_and_dir(
        self, system_pos: np.ndarray, rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SphereShape(ParticleShape):
    """Spawns at center, moves in any direction."""

    def get_spawn_pos_and_dir(
        self, system_pos: np.ndarray, rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:
        phi = rng.uniform(0, 2 * np.pi)
        costheta = rng.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        direction = np.array([x, y, z], dtype=np.float32)
        return system_pos.copy(), direction


class ConeShape(ParticleShape):
    """Spawns at center, moves within a cone angle."""

    def __init__(self, angle_degrees: float = 25.0, direction=(0.0, 1.0, 0.0)):
        self.angle_rad = np.radians(angle_degrees)
        dir_arr = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(dir_arr)
        self.direction = dir_arr / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def get_spawn_pos_and_dir(
        self, system_pos: np.ndarray, rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Sample direction in a cone around +Y axis
        phi = rng.uniform(0, 2 * np.pi)
        z_cone = rng.uniform(np.cos(self.angle_rad), 1.0)
        sin_theta = np.sqrt(1.0 - z_cone**2)
        x = sin_theta * np.cos(phi)
        z = sin_theta * np.sin(phi)
        y = z_cone
        
        local_dir = np.array([x, y, z], dtype=np.float32)
        
        # Rotate local_dir to align with self.direction
        # If direction is +Y, no rotation needed
        if np.allclose(self.direction, [0, 1, 0]):
            return system_pos.copy(), local_dir
        
        # If direction is -Y, flip it
        if np.allclose(self.direction, [0, -1, 0]):
            return system_pos.copy(), np.array([x, -y, z], dtype=np.float32)

        # Standard rotation to align [0, 1, 0] with self.direction
        # Using Rodrigues' rotation formula or similar
        up = np.array([0, 1, 0], dtype=np.float32)
        v = np.cross(up, self.direction)
        c = np.dot(up, self.direction)
        s = np.linalg.norm(v)
        
        if s < 1e-6: # Should be handled by cases above
            return system_pos.copy(), local_dir
            
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
        rotation_matrix = np.eye(3, dtype=np.float32) + kmat + kmat @ kmat * ((1 - c) / (s**2))
        
        final_dir = rotation_matrix @ local_dir
        return system_pos.copy(), final_dir


class BoxShape(ParticleShape):
    """Spawns on one side randomly, moves to the other side."""

    def __init__(self, size=(1.0, 1.0, 1.0), direction=(0.0, 1.0, 0.0)):
        self.size = np.array(size, dtype=np.float32)
        dir_arr = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(dir_arr)
        self.direction = dir_arr / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def get_spawn_pos_and_dir(
        self, system_pos: np.ndarray, rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:
        half = self.size * 0.5
        
        # Determine dominant axis of direction to decide spawn side
        abs_dir = np.abs(self.direction)
        axis = np.argmax(abs_dir)
        sign = np.sign(self.direction[axis])
        if sign == 0: sign = 1.0
        
        # Spawn on the side opposite to the direction sign
        # e.g. if direction is +Y, spawn on -Y side
        pos = np.zeros(3, dtype=np.float32)
        for i in range(3):
            if i == axis:
                pos[i] = -sign * half[i]
            else:
                pos[i] = rng.uniform(-half[i], half[i])
        
        return system_pos + pos, self.direction.copy()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(start: ColorType, end: ColorType, t: float) -> tuple:
    s = np.array(start, dtype=np.float32)
    e = np.array(end, dtype=np.float32)
    if s.max() > 1.0:
        s /= 255.0
    if e.max() > 1.0:
        e /= 255.0
    if len(s) == 3:
        s = np.append(s, 1.0)
    if len(e) == 3:
        e = np.append(e, 1.0)
    return tuple(lerp(s[i], e[i], t) for i in range(4))


def linear_size_over_lifetime(start: float, end: float) -> LifetimeFloat:
    def _curve(t: float) -> float:
        return lerp(start, end, t)
    return _curve


def linear_color_over_lifetime(start: ColorType, end: ColorType) -> LifetimeColor:
    def _curve(t: float) -> ColorType:
        return lerp_color(start, end, t)
    return _curve


def linear_velocity_over_lifetime(start: float, end: float) -> LifetimeVelocity:
    def _curve(t: float):
        return lerp(start, end, t)
    return _curve


class ParticleSystem:
    """Unity-style particle system with pooled Object3D particles."""

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        play_on_awake: bool = True,
        play_duration: float = 0.0,
        particle_life: float = 1.0,
        speed: float = 3.0,
        size: float = 1.0,
        particle_object: Optional[ParticleObject] = None,
        color: Optional[ColorType] = None,
        size_over_lifetime: Optional[LifetimeFloat] = None,
        color_over_lifetime: Optional[LifetimeColor] = None,
        velocity_over_lifetime: Optional[LifetimeVelocity] = None,
        loop: bool = True,
        max_particles: int = 100,
        burst: Optional[ParticleBurst] = None,
        gravity_scale: float = 1.0,
        collider: Optional[Collider] = None,
        shape: Optional[ParticleShape] = None,
    ):
        self._position = np.array(position, dtype=np.float32)
        self.play_on_awake = play_on_awake
        self.play_duration = float(play_duration)
        self.particle_life = float(particle_life)
        self.speed = float(speed)
        self.size = float(size)
        self.particle_object = particle_object
        self.color = color
        self.size_over_lifetime = size_over_lifetime
        self.color_over_lifetime = color_over_lifetime
        self.velocity_over_lifetime = velocity_over_lifetime
        self.loop = loop
        self.max_particles = int(max_particles)
        self.burst = burst or ParticleBurst()
        self.gravity_scale = float(gravity_scale)
        self.collider = collider
        self.shape = shape or SphereShape()

        self._particles: List[Particle] = []
        self._container = None
        self._playing = False
        self._elapsed = 0.0
        self._emit_timer = 0.0
        self._rng = random.Random()

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @position.setter
    def position(self, value):
        self._position = np.array(value, dtype=np.float32)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def _attach(self, container) -> None:
        self._container = container
        self._build_pool()
        if self.play_on_awake:
            self.play()

    def _build_pool(self) -> None:
        if self._container is None:
            raise RuntimeError("ParticleSystem has no container attached.")
        if self._particles:
            return

        for _ in range(self.max_particles):
            obj = self._create_particle_object()
            obj.visible = False
            if self.collider is not None:
                self._attach_collider(obj)
            self._container.add_object(obj)
            self._particles.append(Particle(obj))

    def _create_particle_object(self) -> Object3D:
        if self.particle_object is None:
            obj = create_cube(size=1.0)
        elif isinstance(self.particle_object, Object3D):
            obj = self._clone_object(self.particle_object)
        elif isinstance(self.particle_object, str):
            obj = Object3D(self.particle_object)
        elif callable(self.particle_object):
            obj = self.particle_object()
        else:
            raise ValueError("Unsupported particle_object type")

        obj.scale = self.size
        if self.color is not None:
            obj.color = self.color
        return obj

    def _clone_object(self, template: Object3D) -> Object3D:
        obj = Object3D(position=template.position, scale=template.scale, color=template.color)
        obj.mesh = template.mesh
        obj._mesh_key = template.get_mesh_key()
        obj._uses_texture = getattr(template, "_uses_texture", False)
        obj._texture_image = getattr(template, "_texture_image", None)
        obj._uv = getattr(template, "_uv", None)
        obj._transform_dirty = True
        return obj

    def _attach_collider(self, obj: Object3D) -> Collider:
        template = self.collider
        if template is None:
            raise RuntimeError("ParticleSystem collider template is missing.")

        if isinstance(template, SphereCollider):
            collider = SphereCollider(center=list(template.center), radius=template.radius)
        elif isinstance(template, BoxCollider):
            collider = BoxCollider(center=list(template.center), size=list(template.size))
        else:
            collider = SphereCollider()

        collider.collision_mode = template.collision_mode
        collider.group = template.group
        obj.add_component(collider)
        return collider

    def play(self) -> None:
        self._playing = True
        self._elapsed = 0.0
        self._emit_timer = 0.0

    def stop(self, clear_particles: bool = False) -> None:
        self._playing = False
        if clear_particles:
            for particle in self._particles:
                if particle.active:
                    self._deactivate(particle)

    def emit(self, count: int) -> None:
        if count <= 0:
            return
        for _ in range(count):
            particle = self._get_inactive_particle()
            if particle is None:
                break
            self._activate(particle)

    def update(self, delta_time: float) -> None:
        if self._playing:
            if self.play_duration > 0:
                self._elapsed += delta_time
                if self._elapsed >= self.play_duration:
                    if self.loop:
                        self._elapsed = 0.0
                    else:
                        self._playing = False

            if self._playing:
                self._emit_timer += delta_time
                interval = max(self.burst.interval, 1e-6)
                while self._emit_timer >= interval:
                    self._emit_timer -= interval
                    count = self.burst.count
                    if self.burst.randomize:
                        count = self._rng.randint(0, max(self.burst.count, 0))
                    self.emit(count)

        gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32) * self.gravity_scale
        for particle in self._particles:
            if not particle.active:
                continue

            particle.age += delta_time
            if particle.age >= particle.life:
                self._deactivate(particle)
                continue

            particle.velocity += gravity * delta_time

            life_ratio = particle.age / max(particle.life, 1e-6)
            if self.velocity_over_lifetime is not None:
                vel_value = self.velocity_over_lifetime(life_ratio)
                if isinstance(vel_value, (float, int, np.floating, np.integer)):
                    base = self._normalize_velocity(particle.velocity)
                    particle.velocity = base * float(vel_value)
                else:
                    particle.velocity = np.array(vel_value, dtype=np.float32)

            new_pos = particle.obj.position + particle.velocity * delta_time
            if self.collider is not None:
                self._move_with_collisions(particle, new_pos)
            else:
                particle.obj.position = new_pos

            if self.size_over_lifetime is not None:
                particle.obj.scale = float(self.size_over_lifetime(life_ratio))
            if self.color_over_lifetime is not None:
                particle.obj.color = self.color_over_lifetime(life_ratio)

    def _get_inactive_particle(self) -> Optional[Particle]:
        for particle in self._particles:
            if not particle.active:
                return particle
        return None

    def _activate(self, particle: Particle) -> None:
        particle.active = True
        particle.age = 0.0
        particle.life = self.particle_life

        spawn_pos, spawn_dir = self.shape.get_spawn_pos_and_dir(self._position, self._rng)
        particle.velocity = spawn_dir * self.speed

        particle.obj.visible = True
        particle.obj.position = spawn_pos
        particle.obj.scale = self.size
        if self.color is not None:
            particle.obj.color = self.color
        if self.size_over_lifetime is not None:
            particle.obj.scale = float(self.size_over_lifetime(0.0))
        if self.color_over_lifetime is not None:
            particle.obj.color = self.color_over_lifetime(0.0)
        if self.velocity_over_lifetime is not None:
            vel_value = self.velocity_over_lifetime(0.0)
            if isinstance(vel_value, (float, int, np.floating, np.integer)):
                particle.velocity = self._normalize_velocity(particle.velocity) * float(vel_value)
            else:
                particle.velocity = np.array(vel_value, dtype=np.float32)

    def _deactivate(self, particle: Particle) -> None:
        particle.active = False
        particle.obj.visible = False

    def _normalize_velocity(self, velocity: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(velocity)
        if norm < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return velocity / norm

    def _move_with_collisions(self, particle: Particle, target_pos: np.ndarray) -> None:
        obj = particle.obj
        colliders = obj.get_components(Collider)
        if not colliders:
            obj.position = target_pos
            return

        collider = colliders[0]
        delta = target_pos - obj.position
        if self._container:
            self._container.move_object(obj, delta)
        else:
            obj.position = target_pos

        if collider.collision_mode == CollisionMode.IGNORE:
            obj.position = target_pos
            return

        if np.linalg.norm(obj.position - target_pos) > 1e-6:
            particle.velocity = np.zeros(3, dtype=np.float32)
