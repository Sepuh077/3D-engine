from dataclasses import dataclass
from typing import Optional
from src.engine3d.gameobject import GameObject

@dataclass
class EditorSelection:
    game_object: Optional[GameObject] = None
