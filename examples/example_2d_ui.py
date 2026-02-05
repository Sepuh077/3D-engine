"""
Example: 2D UI overlay with shapes, text and images.
Demonstrates new 2D drawing features (incl. PNG/images) over 3D scene.
"""
import sys
from pathlib import Path
import math
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Demo Arcade-style globals for 2D drawing + get_window
from src.engine3d import (
    Window3D, Keys, Color,
    draw_text, draw_rectangle, draw_circle, draw_ellipse,
    draw_polygon, draw_line, draw_image, get_window,
)
from src.engine3d.object3d import create_cube, create_plane
from src.physics import ColliderType


class UIExample(Window3D):
    """Demo of 2D shapes, text, and UI over 3D."""

    def setup(self):
        """Setup 3D scene and 2D UI state."""
        # Floor
        floor = self.add_object(create_plane(20, 20, color=Color.DARK_GRAY))
        floor.position = (0, 0, 0)
        floor.static = True

        # Some cubes
        for i in range(8):
            cube = self.add_object(create_cube(1.0, color=Color.random_bright()))
            angle = i * (2 * math.pi / 8)
            cube.position = (5 * math.cos(angle), 0.5, 5 * math.sin(angle))
            cube.collider_type = ColliderType.CUBE

        # Player cube
        self.player = self.add_object(create_cube(1.0, color=Color.YELLOW))
        self.player.position = (0, 0.5, 0)
        self.player.collider_type = ColliderType.CUBE

        # Camera
        self.camera.position = (0, 8, 15)
        self.camera.look_at((0, 0, 0))

        # Light
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.4

        # 2D UI state
        self.score = 0
        self.health = 100.0
        self.game_over = False
        self.message = "Collect cubes! (arrow keys)"
        self.time = 0.0

        # Mouse visible for demo
        pygame.mouse.set_visible(True)

        # Generate random image (demo: colored noise pattern, no external file)
        img_array = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        img_array[:, :, 3] = 200  # semi-transparent
        self.random_img = pygame.surfarray.make_surface(img_array[:, :, :3]).convert_alpha()

    def on_update(self, delta_time):
        """Update 3D and UI state."""
        if self.game_over:
            return

        self.time += delta_time
        speed = 10.0 * delta_time

        # Player movement
        moved = False
        if self.is_key_pressed(Keys.LEFT):
            self.player.x -= speed
            moved = True
        if self.is_key_pressed(Keys.RIGHT):
            self.player.x += speed
            moved = True
        if self.is_key_pressed(Keys.UP):
            self.player.z -= speed
            moved = True
        if self.is_key_pressed(Keys.DOWN):
            self.player.z += speed
            moved = True

        # Rotate player
        if moved:
            self.player.rotation_y += 180 * delta_time

        # Fake "collect" by distance to cubes (simple demo)
        for obj in self.objects:
            if obj is self.player or not hasattr(obj, 'position'):
                continue
            dist = math.hypot(
                obj.position[0] - self.player.position[0],
                obj.position[2] - self.player.position[2]
            )
            if dist < 1.5:
                self.score += 10
                # Respawn cube
                angle = self.time * 2
                obj.position = (5 * math.cos(angle), 0.5, 5 * math.sin(angle))
                obj.color = Color.random_bright()

        # Health drain + regen
        self.health = max(0, min(100, self.health - 5 * delta_time + (1 if moved else 5) * delta_time))
        if self.health <= 0:
            self.game_over = True
            self.message = "Game Over! Score: " + str(self.score)

        # Update caption
        self.set_caption(f"Engine3D 2D UI Demo - Score: {self.score} - Health: {int(self.health)} - {self.fps:.0f} FPS")

    def on_draw(self):
        """Draw 2D UI overlay (shapes + text)."""
        if self.game_over:
            # Big centered text
            self.draw_text("GAME OVER", 400, 200, Color.RED, 72, anchor_x='center', anchor_y='center')
            self.draw_text(self.message, 400, 300, Color.WHITE, 36, anchor_x='center')
            self.draw_text("Press R to restart", 400, 380, Color.LIGHT_GRAY, 24, anchor_x='center')
            return

        # Top bar / HUD (demo globals too)
        draw_rectangle(0, 0, 800, 60, (0.1, 0.1, 0.2, 0.8))  # semi-transparent bar
        draw_text(self.message, 20, 10, Color.WHITE, 20)
        draw_text(f"Score: {self.score}", 400, 15, Color.YELLOW, 28, anchor_x='center')

        # Health bar
        bar_w = int(self.health * 3)
        draw_rectangle(20, 45, 300, 12, Color.DARK_GRAY)  # bg
        draw_rectangle(20, 45, bar_w, 12, Color.RED if self.health < 30 else Color.GREEN)
        draw_rectangle(20, 45, 300, 12, Color.WHITE, 2)  # border

        # Timer circle
        self.draw_circle(700, 40, 30, Color.BLUE, border_width=10, aa=False)
        self.draw_circle(700, 40, 30, Color.WHITE, 4, aa=True)
        timer_angle = int((self.time % 10) * 36)  # demo
        # Simple arc sim with polygon (ensure >=3 points)
        pts = [(700, 40)]
        for i in range(0, max(timer_angle, 30) + 1, 10):
            rad = i * math.pi / 180
            pts.append((700 + 25 * math.cos(rad), 40 + 25 * math.sin(rad)))
        self.draw_polygon(pts, Color.CYAN)

        # Demo image (random generated noise)
        draw_image(self.random_img, 650, 450, scale=0.8, alpha=0.9)

        # Bottom status
        self.draw_text("Arrows: Move | ESC: Quit", 20, 550, Color.LIGHT_GRAY, 18)

        # Crosshair
        self.draw_line((395, 300), (405, 300), Color.WHITE, 2)
        self.draw_line((400, 295), (400, 305), Color.WHITE, 2)

    def on_key_press(self, key, modifiers):
        """Handle keys."""
        if key == Keys.ESCAPE:
            self.close()
        elif key == Keys.R and self.game_over:
            # Restart
            self.health = 100
            self.score = 0
            self.game_over = False
            self.message = "Collect cubes! (arrow keys)"
            self.player.position = (0, 0.5, 0)

    def on_mouse_press(self, x, y, button, modifiers):
        """Click to boost health."""
        if button == 1 and not self.game_over:  # left click
            self.health = min(100, self.health + 20)
            self.score += 5


if __name__ == "__main__":
    print("=== Engine3D 2D UI Example ===")
    print("Controls:")
    print("  Arrow Keys - Move player")
    print("  Mouse click - Health boost")
    print("  R - Restart (on game over)")
    print("  ESC - Exit")
    print()
    print("Watch the 2D UI: health bar, score, timer circle, crosshair, random image, etc.")
    print("Supports PNG/image files via draw_image() too.")

    game = UIExample(800, 600, "Engine3D - 2D UI Demo")
    game.run()
