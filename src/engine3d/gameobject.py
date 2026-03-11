from typing import List, Optional, Type, TypeVar, Generator, Any, Dict, Tuple
import importlib
import json
import uuid

import numpy as np

from .component import Component, Script, WaitForSeconds, WaitEndOfFrame, WaitForFrames, Time
from .transform import Transform

T = TypeVar('T', bound=Component)

# Global registry for resolving component references during deserialization
_component_ref_registry: Dict[str, 'GameObject'] = {}

class GameObject:
    def __init__(self, name: str = "GameObject", _id: Optional[str] = None):
        self.name = name
        self.tag = None
        self.components: List[Component] = []
        
        # Unique ID for serialization (auto-generated, not exposed to users)
        self._id = _id if _id else str(uuid.uuid4())
        
        # Every GameObject has a Transform
        self.transform = Transform()
        self.add_component(self.transform)

        # Coroutines state
        self._active_coroutines: List[Dict[str, Any]] = []
        self._end_of_frame_coroutines: List[Dict[str, Any]] = []

    def add_component(self, component: Component) -> Component:
        component.game_object = self
        self.components.append(component)
        component.on_attach()
        return component

    def start_coroutine(self, routine: Generator) -> Generator:
        """Starts a coroutine on this GameObject."""
        # Initial step to get first yield value
        try:
            yield_val = next(routine)
            self._queue_coroutine(routine, yield_val)
        except StopIteration:
            pass
        return routine

    def _queue_coroutine(self, routine: Generator, wait_instruction: Any):
        """Queues a coroutine based on wait instruction."""
        wait = wait_instruction
        # Treat None as wait one frame
        if wait is None:
            wait = WaitForFrames(1)
        
        entry = {
            "generator": routine,
            "wait_instruction": wait
        }
        if isinstance(wait, WaitEndOfFrame):
            self._end_of_frame_coroutines.append(entry)
        else:
            self._active_coroutines.append(entry)

    def _step_coroutines(
        self,
        coroutines: List[Dict[str, Any]],
        delta_time: float,
        allow_end_of_frame: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Advance coroutines and return active + end-of-frame queues."""
        still_active = []
        end_of_frame_queue = []
        for coro in coroutines:
            gen = coro["generator"]
            wait = coro["wait_instruction"]

            is_ready = True
            if isinstance(wait, WaitForSeconds):
                if not wait.is_done(delta_time):
                    is_ready = False
            elif isinstance(wait, WaitForFrames):
                if not wait.step():
                    is_ready = False

            if is_ready:
                try:
                    new_wait = next(gen)
                except StopIteration:
                    continue

                if new_wait is None:
                    new_wait = WaitForFrames(1)

                entry = {
                    "generator": gen,
                    "wait_instruction": new_wait,
                }
                if isinstance(new_wait, WaitEndOfFrame):
                    if allow_end_of_frame:
                        end_of_frame_queue.append(entry)
                    else:
                        still_active.append(entry)
                else:
                    still_active.append(entry)
            else:
                still_active.append(coro)
        
        return still_active, end_of_frame_queue

    def _update_coroutines(self, delta_time: float):
        """Processes active coroutines during the main update phase."""
        self._active_coroutines, new_end_of_frame = self._step_coroutines(
            self._active_coroutines,
            delta_time,
            allow_end_of_frame=True,
        )
        if new_end_of_frame:
            self._end_of_frame_coroutines.extend(new_end_of_frame)

    def _update_end_of_frame_coroutines(self, delta_time: float):
        """Processes coroutines waiting for end of frame."""
        self._end_of_frame_coroutines, deferred_end_of_frame = self._step_coroutines(
            self._end_of_frame_coroutines,
            delta_time,
            allow_end_of_frame=False,
        )
        if deferred_end_of_frame:
            self._end_of_frame_coroutines.extend(deferred_end_of_frame)

    def update(self):
        # Update components
        for comp in self.components:
            comp.update()
        
        # Update coroutines (main phase)
        self._update_coroutines(Time.delta_time)

    def update_end_of_frame(self):
        """Called by Window3D to process end-of-frame coroutines."""
        self._update_end_of_frame_coroutines(Time.delta_time)

    def start_scripts(self):
        """Call awake() and start() on all Script components that haven't been started yet."""
        for comp in self.components:
            if isinstance(comp, Script):
                # Call awake() first if not already awoken
                if not comp._awoken:
                    comp.awake()
                    comp._awoken = True
                # Then call start() if not already started
                if not comp._started:
                    comp.start()
                    comp._started = True

    def awake_scripts(self):
        """Call awake() on all Script components that haven't been awoken yet."""
        for comp in self.components:
            if isinstance(comp, Script) and not comp._awoken:
                comp.awake()
                comp._awoken = True

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        for comp in self.components:
            if isinstance(comp, component_type):
                return comp
        return None

    def get_components(self, component_type: Type[T]) -> List[T]:
        return [comp for comp in self.components if isinstance(comp, component_type)]

    def __repr__(self):
        return f"GameObject(name='{self.name}', position={self.transform.position})"

    # =========================================================================
    # Prefab serialization
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Save this GameObject (components + start values) to a prefab file.
        """
        data = self._to_prefab_dict()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
    ) -> "GameObject":
        """
        Load a GameObject prefab from disk and return the created GameObject.
        
        Args:
            path: Path to the prefab file
            position: Optional position to set (x, y, z). Defaults to (0, 0, 0).
            rotation: Optional rotation to set (rx, ry, rz) in degrees. Defaults to (0, 0, 0).
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        obj = cls._from_prefab_dict(data)
        
        # Apply position and rotation
        if position is not None:
            obj.transform.position = position
        if rotation is not None:
            obj.transform.rotation = rotation
        
        return obj

    def _to_prefab_dict(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "name": self.name,
            "tag": self.tag,
            "components": [self._component_to_prefab(comp, idx) for idx, comp in enumerate(self.components)],
        }

    @classmethod
    def _from_prefab_dict(cls, data: Dict[str, Any]) -> "GameObject":
        game_object = cls(name=data.get("name", "GameObject"), _id=data.get("_id"))
        game_object.tag = data.get("tag")

        components_data = data.get("components", [])
        for comp_data in components_data:
            component = cls._component_from_prefab(comp_data)
            if component is None:
                continue
            if isinstance(component, Transform):
                game_object.transform = component
                game_object.components[0] = component
                component.game_object = game_object
            else:
                game_object.add_component(component)

        return game_object

    @staticmethod
    def _component_to_prefab(component: Component, component_index: int = 0) -> Dict[str, Any]:
        comp_cls = component.__class__
        module_name = comp_cls.__module__
        class_name = comp_cls.__name__

        skip_keys = {
            "game_object",
            "_mesh",
            "mesh",
            "_vao",
            "_vbo",
            "_gl_texture",
            "_gpu_initialized",
            "_mesh_key",
            "_mesh_cache",
            "_texture_image",
        }

        is_object3d = module_name == "src.engine3d.object3d" and class_name == "Object3D"
        if is_object3d:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_local_min",
                "_local_max",
                "_local_radius",
                "_uv",
            })

        is_collider = module_name.startswith("src.physics")
        if is_collider:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_current_collisions",
                "mesh_data",
                "sphere",
                "obb",
                "aabb",
                "cylinder",
                "_transform_dirty",
            })

        # Get the game object ID for component references
        game_object_id = component.game_object._id if component.game_object else None

        state = {
            key: GameObject._serialize_value(value, game_object_id, component_index)
            for key, value in component.__dict__.items()
            if key not in skip_keys
        }

        data = {
            "module": module_name,
            "class": class_name,
            "state": state,
        }
        return data

    @staticmethod
    def _component_from_prefab(data: Dict[str, Any]) -> Optional[Component]:
        module_name = data.get("module")
        class_name = data.get("class")
        state = data.get("state", {})
        if not module_name or not class_name:
            return None

        module = importlib.import_module(module_name)
        comp_cls = getattr(module, class_name, None)
        if comp_cls is None:
            raise ValueError(f"Component class '{class_name}' not found in {module_name}")

        component: Component = comp_cls()
        
        # First pass: deserialize without resolving component refs
        # Store the raw state for later resolution
        restored_state = GameObject._deserialize_value(state, None)  # None = no registry yet
        component.__dict__.update(restored_state)
        
        # Store the raw serialized state for second-pass resolution
        component._serialized_state = state

        if module_name == "src.engine3d.object3d" and class_name == "Object3D":
            GameObject._restore_object3d_geometry(component)

        return component

    @staticmethod
    def _serialize_value(value: Any, source_go_id: str = None, source_comp_idx: int = 0) -> Any:
        # Handle component references
        if isinstance(value, Component):
            if value.game_object:
                # Find the component index
                comp_idx = -1
                for idx, comp in enumerate(value.game_object.components):
                    if comp is value:
                        comp_idx = idx
                        break
                if comp_idx >= 0:
                    return {
                        "__type__": "component_ref",
                        "game_object_id": value.game_object._id,
                        "component_index": comp_idx,
                        "component_class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                    }
            return None  # Component without game_object, can't reference
        
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(value.dtype),
                "value": value.tolist(),
            }
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            return value.item()
        try:
            from src.physics.group import ColliderGroup
        except ImportError:
            ColliderGroup = None
        if ColliderGroup is not None and isinstance(value, ColliderGroup):
            return {
                "__type__": "ColliderGroup",
                "name": value.name,
            }

        try:
            from src.engine3d.graphics.material import Material
        except ImportError:
            Material = None
        if Material is not None and isinstance(value, Material):
            state = {
                k: GameObject._serialize_value(v, source_go_id, source_comp_idx)
                for k, v in value.__dict__.items()
            }
            return {
                "__type__": "Material",
                "class": value.__class__.__name__,
                "state": state,
            }

        if isinstance(value, dict):
            return {key: GameObject._serialize_value(val, source_go_id, source_comp_idx) for key, val in value.items()}
        if isinstance(value, list):
            return [GameObject._serialize_value(val, source_go_id, source_comp_idx) for val in value]
        if isinstance(value, set):
            return {
                "__type__": "set",
                "value": [GameObject._serialize_value(val, source_go_id, source_comp_idx) for val in value],
            }
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "value": [GameObject._serialize_value(val, source_go_id, source_comp_idx) for val in value],
            }
        if isinstance(value, bytes):
            return {
                "__type__": "bytes",
                "value": list(value),
            }
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            return {
                "__type__": "repr",
                "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "value": repr(value),
            }
        return value

    @staticmethod
    def _deserialize_value(value: Any, go_registry: Dict[str, 'GameObject'] = None) -> Any:
        if isinstance(value, dict):
            # Handle component references
            if value.get("__type__") == "component_ref":
                if go_registry is None:
                    return None  # Can't resolve without registry
                go_id = value.get("game_object_id")
                comp_idx = value.get("component_index", 0)
                go = go_registry.get(go_id)
                if go and 0 <= comp_idx < len(go.components):
                    return go.components[comp_idx]
                return None
            
            if value.get("__type__") == "ndarray":
                return np.array(value.get("value", []), dtype=value.get("dtype", None))
            if value.get("__type__") == "tuple":
                return tuple(GameObject._deserialize_value(val, go_registry) for val in value.get("value", []))
            if value.get("__type__") == "set":
                return set(GameObject._deserialize_value(val, go_registry) for val in value.get("value", []))
            if value.get("__type__") == "ColliderGroup":
                from src.physics.group import ColliderGroup
                name = value.get("name", "default")
                return ColliderGroup._registry.get(name) or ColliderGroup(name)
            if value.get("__type__") == "Material":
                from src.engine3d.graphics import material
                class_name = value.get("class", "LitMaterial")
                state = value.get("state", {})
                mat_cls = getattr(material, class_name, material.LitMaterial)
                mat = mat_cls()
                restored_state = GameObject._deserialize_value(state, go_registry)
                mat.__dict__.update(restored_state)
                return mat
            if value.get("__type__") == "bytes":
                return bytes(value.get("value", []))
            if value.get("__type__") == "repr":
                return value.get("value")
            return {key: GameObject._deserialize_value(val, go_registry) for key, val in value.items()}
        if isinstance(value, list):
            return [GameObject._deserialize_value(val, go_registry) for val in value]
        return value

    @staticmethod
    def _restore_object3d_geometry(component: Component) -> None:
        try:
            from src.engine3d.object3d import Object3D
        except ImportError:
            return
        if not isinstance(component, Object3D):
            return

        source_type = getattr(component, "_source_type", "none")
        if source_type == "file":
            source_path = getattr(component, "_source_path", None)
            if source_path:
                component.load(source_path)
        elif source_type == "primitive":
            prim_type = getattr(component, "_primitive_type", None)
            params = getattr(component, "_primitive_params", {}) or {}
            if prim_type == "cube":
                size = params.get("size", 1.0)
                from src.engine3d.object3d import create_cube
                temp_go = create_cube(size)
                temp_obj = temp_go.get_component(Object3D)
                if temp_obj:
                    component.mesh = temp_obj.mesh
                    component._post_process_geometry(f"primitive_cube_{size}")
            elif prim_type == "sphere":
                radius = params.get("radius", 1.0)
                subdivisions = params.get("subdivisions", 2)
                from src.engine3d.object3d import create_sphere
                temp_go = create_sphere(radius, subdivisions=subdivisions)
                temp_obj = temp_go.get_component(Object3D)
                if temp_obj:
                    component.mesh = temp_obj.mesh
                    component._post_process_geometry(f"primitive_sphere_{radius}")
            elif prim_type == "plane":
                width = params.get("width", 10.0)
                height = params.get("height", 10.0)
                from src.engine3d.object3d import create_plane
                temp_go = create_plane(width, height)
                temp_obj = temp_go.get_component(Object3D)
                if temp_obj:
                    component.mesh = temp_obj.mesh
                    component._post_process_geometry(f"primitive_plane_{width}_{height}")
