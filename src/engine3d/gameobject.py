from typing import List, Optional, Type, TypeVar
import numpy as np

from .component import Component, Transform

T = TypeVar('T', bound=Component)

class GameObject:
    def __init__(self, name: str = "GameObject"):
        self.name = name
        self.tag = None
        self.components: List[Component] = []
        
        # Every GameObject has a Transform
        self.transform = Transform()
        self.add_component(self.transform)

    def add_component(self, component: Component) -> Component:
        component.game_object = self
        self.components.append(component)
        component.on_attach()
        return component

    def update(self, delta_time: float):
        for comp in self.components:
            comp.update(delta_time)

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        for comp in self.components:
            if isinstance(comp, component_type):
                return comp
        return None

    def get_components(self, component_type: Type[T]) -> List[T]:
        return [comp for comp in self.components if isinstance(comp, component_type)]

    def __repr__(self):
        return f"GameObject(name='{self.name}', position={self.transform.position})"
