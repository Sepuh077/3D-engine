"""
Example game demonstrating the new ObjectGroup collision system.
Tests ignore, detect-pass-through (triggers), and solid (block) groups.
Also demonstrates OnCollisionEnter/Exit/Stay callbacks.
"""
import os
import sys
import math
import pygame

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from src.engine3d import Window3D, Keys, Color
from src.engine3d.object3d import create_cube, create_plane, Object3D
from src.physics import CollisionMode, BoxCollider, SphereCollider, CapsuleCollider, Collider


# Player uses custom OnCollision* (now on collider; other is Collider, main obj via .object3d)
def make_player_callbacks(player):
    player_coll = player.get_component(Collider)
    def on_enter(other):
        # other is other Collider; get main object
        other_obj = other.object3d
        if hasattr(other_obj, 'color_on_trigger'):
            other_obj.color = other_obj.color_on_trigger
        print(f"Player entered collision with {other_obj.name or 'obj'}")
    def on_exit(other):
        other_obj = other.object3d
        if hasattr(other_obj, 'color_normal'):
            other_obj.color = other_obj.color_normal
        print(f"Player exited collision with {other_obj.name or 'obj'}")
    def on_stay(other):
        other_obj = other.object3d
        # Use name/layer (masks replace groups)
        if getattr(other_obj, 'name', '') == "Wall" or getattr(other_obj, 'name', '') == "Floor":
            print(f"Player Stayed ---------------- with {other_obj.name or 'obj'}")
    if player_coll:
        player_coll.OnCollisionEnter = on_enter
        player_coll.OnCollisionExit = on_exit
        player_coll.OnCollisionStay = on_stay


class CollisionGroupsExample(Window3D):
    """Tests layer/mask collision filtering (replaces groups)."""
    
    def setup(self):
        # Floor (solid with player; add collider separately)
        floor = self.add_object(create_plane(30, 30, color=Color.DARK_GRAY))
        floor.position = (0, -0.5, 0)
        floor.static = True
        floor.name = "Floor"
        fcoll = floor.add_component(BoxCollider())
        fcoll.collision_mode = CollisionMode.NORMAL
        fcoll.layer = 1
        fcoll.collision_mask = 0xffffffff
        
        # Walls (solid)
        self.walls = []
        wall_positions = [(-10, 1, 0), (10, 1, 0), (0, 1, -10), (0, 1, 10)]
        for pos in wall_positions:
            wall = self.add_object(create_cube(2.0, color=Color.GRAY))
            wall.position = pos
            wall.static = True
            wall.name = "Wall"
            wcoll = wall.add_component(BoxCollider())
            wcoll.collision_mode = CollisionMode.NORMAL
            wcoll.layer = 1
            wcoll.collision_mask = 0xffffffff
            self.walls.append(wall)
        
        # Trigger objects (pass through, change color on contact)
        self.triggers = []
        trigger_pos = [(-5, 1, 5), (5, 1, 5)]
        for i, pos in enumerate(trigger_pos):
            trig = self.add_object(create_cube(1.5, color=Color.YELLOW))
            trig.position = pos
            trig.name = f"Trigger{i}"
            trig.color_normal = Color.YELLOW
            trig.color_on_trigger = Color.PURPLE
            tcoll = trig.add_component(SphereCollider())
            tcoll.collision_mode = CollisionMode.TRIGGER  # detect but pass
            tcoll.layer = 2
            tcoll.collision_mask = 0xffffffff
            self.triggers.append(trig)
        
        # Ignore objects (can overlap freely, no events)
        self.ignores = []
        ignore_pos = [(-5, 1, -5), (5, 1, -5)]
        for i, pos in enumerate(ignore_pos):
            ign = self.add_object(create_cube(1.5, color=Color.ORANGE))
            ign.position = pos
            ign.name = f"Ignore{i}"
            icoll = ign.add_component(BoxCollider())
            icoll.collision_mode = CollisionMode.IGNORE
            icoll.layer = 3
            icoll.collision_mask = 0  # ignore all
            self.ignores.append(ign)
        
        # Player (cube collider + mesh; user adds collider separately)
        player_base = create_cube(1.0, color=Color.BLUE)
        self.player = self.add_object(player_base)
        self.player.scale = 1.0
        self.player.position = (0, 0.5, 0)
        self.player.name = "Player"
        self.player.move_speed = 100.0
        # Add collider with mode/mask (masks replace groups/relations)
        pcoll = self.player.add_component(BoxCollider())
        pcoll.collision_mode = CollisionMode.CONTINUOUS
        pcoll.layer = 10
        pcoll.collision_mask = 0xffffffff  # collide all
        self.player.collision_modes = [CollisionMode.NORMAL, CollisionMode.CONTINUOUS, CollisionMode.IGNORE]
        self.player.mode_idx = 1
        # Attach callbacks
        make_player_callbacks(self.player)
        
        # Camera
        self.camera.position = (0, 15, 20)
        self.camera.look_at((0, 0, 0))
        
        # Light
        self.light.direction = (0.5, -0.8, -0.5)
        self.light.ambient = 0.4
        
        # UI state
        self.show_colliders = True
        self.collision_count = 0
    
    def on_update(self, delta_time):
        # Player movement with WASD + arrows for Y
        dx = dy = dz = 0.0
        speed = self.player.move_speed * delta_time
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dz -= speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dz += speed
        if keys[pygame.K_SPACE]:
            dy += speed
        if keys[pygame.K_LSHIFT]:
            dy -= speed
        
        if dx or dy or dz:
            self.move_object(self.player, (dx, dy, dz))
        
        # Count active collisions for display (from collider)
        pcoll = self.player.get_component(Collider)
        self.collision_count = len(pcoll._current_collisions) if pcoll else 0
        
        # Update caption (mode from collider)
        pos = self.player.position
        pcoll = self.player.get_component(Collider)
        mode_str = str(pcoll.collision_mode).split('.')[-1] if pcoll else "N/A"
        self.set_caption(
            f"Groups Demo - Player: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
            f"Mode:{mode_str} Speed:{self.player.move_speed} | "
            f"Collisions: {self.collision_count} | "
            f"FPS: {self.fps:.0f}"
        )
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.close()
        elif key == Keys.SPACE:
            self.show_colliders = not self.show_colliders
        elif key == pygame.K_c:
            # Cycle collision mode (on collider)
            pcoll = self.player.get_component(Collider)
            if pcoll:
                self.player.mode_idx = (self.player.mode_idx + 1) % len(self.player.collision_modes)
                pcoll.collision_mode = self.player.collision_modes[self.player.mode_idx]
        elif key == pygame.K_1:
            self.player.move_speed = 10.0
        elif key == pygame.K_2:
            self.player.move_speed = 100.0
        elif key == pygame.K_3:
            self.player.move_speed = 1000.0
    
    def on_draw(self):
        # Draw colliders if enabled
        if self.show_colliders:
            for obj in self.objects:
                # Color by group (legacy; now use masks)
                col = Color.WHITE
                if obj == self.player:
                    col = Color.BLUE
                elif hasattr(obj, 'name') and 'Wall' in obj.name:
                    col = Color.RED
                elif hasattr(obj, 'name') and 'Trigger' in obj.name:
                    col = Color.PURPLE
                self.draw_collider(obj, col)
        # Show mode/speed info
        pcoll = self.player.get_component(Collider)
        mode_str = str(pcoll.collision_mode).split('.')[-1] if pcoll else "N/A"
        self.draw_text(f"Mode: {mode_str} | Speed: {self.player.move_speed}", 10, 10, Color.WHITE, 20)
        # Note: on_draw can add 2D UI if needed


if __name__ == "__main__":
    print("=== ObjectGroup Collision System Demo ===")
    print("Controls:")
    print("  WASD/Arrows - Move player")
    print("  1/2/3 - Set speed (10/100/1000)")
    print("  C - Cycle collision mode (test fast/ignore)")
    print("  SPACE - Toggle colliders")
    print("  ESC - Quit")
    print()
    print("Groups:")
    print("  Blue player (cube): collides with walls (red, blocks), triggers (purple, pass-thru), ignores orange")
    print("  Red walls: solid, block player")
    print("  Purple triggers: pass through, change color on contact, OnCollision* called")
    print("  Orange ignores: overlap freely, no detection/events")
    print()
    print("Watch console for Enter/Exit prints and color changes.")
    print()
    game = CollisionGroupsExample(900, 600, "Engine3D - ObjectGroup Collision Demo")
    game.run()
