"""
Example: First-person camera controls
Demonstrates FPS-style camera movement with mouse look.
"""
import sys
from pathlib import Path
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random

from src.engine3d import Window3D, Keys, Color
from src.engine3d.object3d import create_cube, create_plane
from src.physics.collision import objects_collide
from src.physics import ColliderType


class FPSCameraExample(Window3D):
    """First-person camera example."""
    
    def setup(self):
        # Create a floor
        floor = self.add_object(create_plane(50, 50, color=Color.DARK_GRAY))
        floor.position = (0, 0, 0)
        # floor.static = True
        values = ColliderType.all()
        
        # Create some objects to look at
        for x in range(-40, 41, 4):
            for z in range(-40, 41, 4):
                if x == 0 and z == 0:
                    continue
                cube = self.add_object(create_cube(1.0, color=Color.random_bright(), collider_type=random.choice(values)))
                if random.random() < 0.5:
                    cube.static = True
                cube.position = (x, 0.5, z)
        
        # Create taller pillars
        for i in range(4):
            pillar = self.add_object(create_cube(2.0, color=Color.BLUE, collider_type=random.choice(values)))
            angle = i * math.pi / 2
            pillar.position = (
                15 * math.cos(angle),
                2,
                15 * math.sin(angle)
            )
            pillar.scale_xyz = (2, 4, 2)
        
        # Load the stairs model
        stairs = self.load_object(
            "example/stairs_modular_right.obj",
            position=(0, 0, 0),
            scale=2.0,
            color=Color.ORANGE
        )
        
        # Camera setup - first person style
        self.camera.position = (0, 2, 10)
        self.camera.look_at((0, 2, 0))
        self.camera_obj = create_cube(1, self.camera.position)
        
        # Mouse look settings
        self.mouse_sensitivity = 0.002
        self.move_speed = 10.0
        
        # Hide mouse cursor and capture it
        import pygame
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.yaw = 0
        self.pitch = 0
    
    def on_update(self, delta_time):        
        # Movement
        speed = self.move_speed * delta_time
        
        if self.is_key_pressed(Keys.W):
            self.camera.move_forward(speed)
        if self.is_key_pressed(Keys.S):
            self.camera.move_forward(-speed)
        if self.is_key_pressed(Keys.A):
            self.camera.move_right(-speed)
        if self.is_key_pressed(Keys.D):
            self.camera.move_right(speed)
        if self.is_key_pressed(Keys.SPACE):
            self.camera.move_up(speed)
        if self.is_key_pressed(Keys.LSHIFT):
            self.camera.move_up(-speed)

        self.camera_obj.position = self.camera.position
        
        # Rotate all cubes
        for obj in self.objects:
            if not obj.static:
                obj.rotation_y += delta_time * 20
                obj.rotation_x += delta_time * 10
                obj.rotation_z += delta_time * 5
            if objects_collide(obj, self.camera_obj):
                print(obj)
        
        # Update title
        pos = self.camera.position
        self.set_caption(f"FPS Camera - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - {self.fps:.0f} FPS")
    
    def on_mouse_motion(self, x, y, dx, dy):
        import math
        import numpy as np
        
        # Update yaw and pitch
        self.yaw -= dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        
        # Clamp pitch
        self.pitch = max(-1.5, min(1.5, self.pitch))
        
        # Calculate new look direction
        look_x = -math.cos(self.pitch) * math.sin(self.yaw)
        look_y = math.sin(self.pitch)
        look_z = -math.cos(self.pitch) * math.cos(self.yaw)
        
        # Update camera target
        pos = self.camera.position
        self.camera.target = (
            pos[0] + look_x,
            pos[1] + look_y,
            pos[2] + look_z
        )
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            import pygame
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            self.close()

    def on_draw(self):
        for obj in self.objects:
            obj.draw_collider(self, color=(0, 1, 0))


if __name__ == "__main__":
    print("=== Engine3D FPS Camera Example ===")
    print("Controls:")
    print("  WASD - Move")
    print("  SPACE - Move up")
    print("  SHIFT - Move down")
    print("  Mouse - Look around")
    print("  ESC - Exit")
    print()
    
    game = FPSCameraExample(800, 600, "Engine3D - FPS Camera")
    game.run(200)
