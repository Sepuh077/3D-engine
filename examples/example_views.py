"""
Example: Using Views for scene management
Demonstrates View3D for creating menu, game, and pause screens.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine3d import Window3D, View3D, Object3D, Keys, Color
from src.engine3d.object3d import create_cube


class MenuView(View3D):
    """Main menu scene."""
    
    def setup(self):
        print("Setting up Menu...")
        
        # Create some cubes as menu decoration
        self.title_cube = self.add_object(create_cube(2.0, color=Color.BLUE))
        self.title_cube.transform.position = (0, 2, 0)
        
        # Camera looking at scene
        self.camera.position = (0, 5, 15)
        self.camera.look_at((0, 2, 0))
        
        self.time = 0
    
    def on_update(self, delta_time):
        self.time += delta_time
        # Gentle rotation
        self.title_cube.transform.rotation_y = self.time * 20
        self.title_cube.y = 2 + 0.5 * (1 + __import__('math').sin(self.time * 2))
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ENTER or key == Keys.SPACE:
            # Switch to game view
            print("Starting game...")
            self.window.show_view(GameView())
        elif key == Keys.ESCAPE:
            self.window.close()


class GameView(View3D):
    """Main game scene."""
    
    def setup(self):
        print("Setting up Game...")
        
        # Load player object
        self.player = self.load_object(
            "example/stairs_modular_right.obj",
            position=(0, 0, 0),
            scale=1.0,
            color=Color.ORANGE
        )
        
        # Create some environment objects
        for i in range(5):
            cube = self.add_object(create_cube(1.0, color=Color.GREEN))
            cube.transform.position = (-8 + i * 4, 0, -5)
            cube.tag = "obstacle"
        
        # Camera
        self.camera.position = (0, 10, 20)
        self.camera.look_at((0, 0, 0))
        
        # Light
        self.light.direction = (0.5, -0.8, -0.3)
        
        # Game state
        self.paused = False
        self.score = 0
    
    def on_update(self, delta_time):
        if self.paused:
            return
        
        # Player movement
        speed = 5 * delta_time
        
        if self.window.is_key_pressed(Keys.W):
            self.player.z -= speed
        if self.window.is_key_pressed(Keys.S):
            self.player.z += speed
        if self.window.is_key_pressed(Keys.A):
            self.player.x -= speed
        if self.window.is_key_pressed(Keys.D):
            self.player.x += speed
        
        # Rotate player
        self.player.transform.rotation_y += delta_time * 30
        
        # Camera follows player
        px, py, pz = self.player.transform.position
        self.camera.target = (px, py, pz)
        self.camera.position = (px, py + 10, pz + 20)
        
        # Rotate obstacles
        for obj in self.get_objects_by_tag("obstacle"):
            obj.transform.rotation_y += delta_time * 50
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            # Go back to menu
            print("Returning to menu...")
            self.window.show_view(MenuView())
        elif key == Keys.P:
            # Toggle pause
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")


class MyGame(Window3D):
    """Main application."""
    
    def setup(self):
        # Start with menu view
        self.show_view(MenuView())


if __name__ == "__main__":
    print("=== Engine3D Views Example ===")
    print("Controls:")
    print("  ENTER/SPACE - Start game (from menu)")
    print("  WASD - Move player (in game)")
    print("  P - Pause/Resume (in game)")
    print("  ESC - Back to menu / Exit")
    print()
    
    game = MyGame(800, 600, "Engine3D - Views Example")
    game.run()
