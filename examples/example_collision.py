"""
Example: Collision Detection and Bounding Boxes
Demonstrates collision detection between objects and visual bounding boxes.
"""
import os
import sys
import math
import numpy as np
import pygame
import random

# Add the project root to sys.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from src.engine3d import Window3D, Keys, Color
from src.engine3d.object3d import create_cube, create_plane
from src.physics import BoxCollider, SphereCollider, Collider


class CollisionExample(Window3D):
    """Example demonstrating collision detection and bounding boxes."""

    def setup(self):
        """Called once at startup."""
        # Create some static obstacles
        floor = self.add_object(create_plane(50, 50, color=Color.DARK_GRAY))
        floor.position = (0, 0, 0)
        floor.static = True
        floor.add_component(BoxCollider())  # user adds

        self.obstacles = []
        # 2 cube obstacles, 2 sphere obstacles
        positions = [
            (-5, 1, 0),
            (5, 1, 0),
            (0, 1, -5),
            (0, 1, 5)
        ]
        colliders = [BoxCollider, BoxCollider, SphereCollider, SphereCollider]
        for i in range(4):
            obs = self.add_object(create_cube(2.0, color=Color.GREEN))
            obs.position = positions[i]
            obs.add_component(colliders[i]())  # user adds
            self.obstacles.append(obs)

        # Create a moving player object (sphere collider for testing sphere-sphere)
        self.player = self.add_object(create_cube(1.0, color=Color.BLUE))
        self.player.position = (0, 0.5, 0)
        self.player.add_component(SphereCollider())  # user adds
        self.player.impassable_objects.extend(self.obstacles)
        self.player.impassable_objects.append(floor)

        # Create some moving enemies (use stairs OBJ for reliability; GLTF needs .bin)
        self.enemies = []
        obj_path = r"D:\workspace\3d-game-engine\assets\glTF\Bush_Common_Flowers.gltf"
        for i in range(2):
            enemy = self.add_object(obj_path, scale=1, color=Color.RED)  # color tints if no vertex colors
            enemy.position = (-3 + i * 6, 0.75, -3 + i * 6)
            enemy.speed = 2.0 + i * 0.5  # Slower movement for visibility
            self.enemies.append(enemy)

        # Set up camera
        self.camera.position = (0, 15, 15)
        self.camera.look_at((0, 0, 0))

        # Set up light
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.3

        # Movement speed
        self.move_speed = 10.0  # Increased for more visible movement

        # Toggle for bounding boxes
        self.show_bounding_boxes = True

        # Movement state
        self.move_dir = [0, 0, 0]  # x, y, z

    def on_update(self, delta_time):
        """Called every frame."""
        # Move enemies in circles
        if not hasattr(self, 'time_elapsed'):
            self.time_elapsed = 0.0
        self.time_elapsed += delta_time

        for i, enemy in enumerate(self.enemies):
            angle = self.time_elapsed * enemy.speed + i * math.pi / 2
            radius = 3.0
            enemy.x = math.cos(angle) * radius
            enemy.z = math.sin(angle) * radius
            enemy.y = 0.75

            # Check collision with player (via colliders)
            pcoll = self.player.get_component(Collider)
            ecoll = enemy.get_component(Collider)
            if pcoll and ecoll and pcoll.check_collision(ecoll):
                # Collision detected - change color or something
                enemy._color = np.array(Color.YELLOW, dtype=np.float32)
            else:
                enemy._color = np.array(Color.RED, dtype=np.float32)

        # Update movement direction
        keys = pygame.key.get_pressed()
        self.move_dir = [0, 0, 0]
        if keys[pygame.K_w]:
            self.move_dir[2] = -1
        if keys[pygame.K_s]:
            self.move_dir[2] = 1
        if keys[pygame.K_a]:
            self.move_dir[0] = -1
        if keys[pygame.K_d]:
            self.move_dir[0] = 1
        self.move_dir[1] = -0.1

        # Move player based on input
        delta = self.move_speed * delta_time
        dx = self.move_dir[0] * delta
        dy = self.move_dir[1] * delta
        dz = self.move_dir[2] * delta
        if dx != 0 or dy != 0 or dz != 0:
            self.move_object(self.player, (dx, dy, dz))

        # Update window title
        pos = self.player.position
        self.set_caption(
            f"Collision Demo - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - "
            f"BBoxes: {'ON' if self.show_bounding_boxes else 'OFF'} - {self.fps:.0f} FPS"
        )

    def on_key_press(self, key, modifiers):
        """Called when a key is pressed."""
        if key == Keys.ESCAPE:
            self.close()
        elif key == Keys.SPACE:
            # Toggle bounding boxes
            self.show_bounding_boxes = not self.show_bounding_boxes
        elif key == Keys.R:
            # Reset player position
            self.player.position = (0, 0.5, 0)

    def on_draw(self):
        if self.show_bounding_boxes:
            for obj in self.objects:
                self.draw_collider(obj, Color.WHITE)
        return super().on_draw()


if __name__ == "__main__":
    print("=== Engine3D Collision Detection Example ===")
    print("Controls:")
    print("  WASD - Move blue cube (player)")
    print("  SPACE - Toggle bounding boxes")
    print("  R - Reset player position")
    print("  ESC - Exit")
    print()
    print("IMPORTANT: Click on the window to focus it, then use keyboard controls.")
    print()
    print("Green: Static obstacles (2 cubes + 2 spheres)")
    print("Red/Yellow stairs: Moving enemies (yellow when colliding with player)")
    print("Blue: Player (sphere collider) - moves with WASD (cannot pass through obstacles)")
    print("White lines: Bounding boxes (toggle with SPACE)")
    print()

    game = CollisionExample(800, 600, "Engine3D - Collision Demo")
    game.run()