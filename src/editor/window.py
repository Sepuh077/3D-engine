from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable, Any, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

from src.engine3d.scene import Scene3D
from src.engine3d.window import Window3D
from src.engine3d.gameobject import GameObject
from src.engine3d.object3d import create_cube, create_sphere, create_plane, Object3D
from src.engine3d.component import InspectorField, InspectorFieldType

from .selection import EditorSelection
from .viewport import ViewportWidget
from .scene import EditorScene


class HierarchyTreeWidget(QtWidgets.QTreeWidget):
    """Custom tree widget that supports drag-drop parenting of GameObjects."""
    object_parented = QtCore.Signal(object, object)  # (child_obj, parent_obj or None)
    
    def __init__(self, editor_window, parent=None):
        super().__init__(parent)
        self.editor_window = editor_window
        self._dragged_item = None
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
    
    def startDrag(self, supported_actions) -> None:
        self._dragged_item = self.currentItem()
        super().startDrag(supported_actions)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle drop event to parent objects."""
        # Get the item being dragged
        dragged_item = self._dragged_item or self.currentItem()
        if not dragged_item:
            return
        
        # Get the drop target
        drop_item = self.itemAt(event.position().toPoint())
        
        # Find the GameObjects from items
        dragged_obj = None
        drop_obj = None
        
        for obj, item in self.editor_window._object_items.items():
            if item is dragged_item:
                dragged_obj = obj
            if item is drop_item:
                drop_obj = obj
        
        if not dragged_obj:
            return
        
        # Check for circular parenting (can't drop parent onto its child)
        if drop_obj and self._is_descendant(dragged_obj, drop_obj):
            return  # Invalid drop

        # Allow dropping onto viewport or empty area to unparent
        if drop_item is None:
            drop_obj = None
        
        # Emit signal for the parenting operation
        # If drop_obj is None, it means dropping at root level
        self.object_parented.emit(dragged_obj, drop_obj)
        
        # Accept the event
        event.acceptProposedAction()
        self._dragged_item = None
    
    def _is_descendant(self, potential_ancestor: GameObject, potential_descendant: GameObject) -> bool:
        """Check if potential_descendant is a descendant of potential_ancestor."""
        current = potential_descendant.transform.parent
        while current:
            if current.game_object is potential_ancestor:
                return True
            current = current.parent
        return False


class EditorWindow(QtWidgets.QMainWindow):
    # Signal emitted when a play mode error occurs
    play_mode_error = QtCore.Signal(str, str)  # (error_message, traceback_text)
    
    def __init__(self, project_root: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.setWindowTitle("Engine3D Editor")
        self.resize(1280, 768)

        self._selection = EditorSelection()
        self._scene = EditorScene()
        self._window: Optional[Window3D] = None
        self._scene_auto_objects = set() # Show all objects
        self._object_items: Dict[GameObject, QtWidgets.QTreeWidgetItem] = {}
        self._component_fields: list[QtWidgets.QWidget] = []
        self._components_dirty = True

        # Scene file management
        self._current_scene_path: Optional[Path] = None
        self._scene_dirty = False
        self._scene_name = "Untitled Scene"

        # Editor camera (separate from game camera)
        from src.engine3d.camera import Camera3D
        self._editor_camera = Camera3D()

        # Play mode state
        self._playing = False
        self._paused = False
        self._original_scene_data = None

        # Camera control state
        self._camera_control = {
            'orbiting': False,
            'panning': False,
            'last_mouse_pos': None,
            'azimuth': 45.0,  # Horizontal angle around target
            'elevation': 45.0,  # Vertical angle
            'distance': 10.0,  # Distance from target
            'target': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

        # File watcher for code changes (hot reload)
        self._file_watcher = QtCore.QFileSystemWatcher(self)
        self._file_watcher.fileChanged.connect(self._on_script_file_changed)
        self._watched_script_files: Dict[str, float] = {}  # path -> last modified time
        self._script_reload_pending = False
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._reload_script_components)

        # Connect play mode error signal
        self.play_mode_error.connect(self._on_play_mode_error)

        self._build_layout()
        self._setup_files_panel()
        self._setup_hierarchy_panel()
        self._setup_inspector_panel()
        self._setup_toolbar()
        self._setup_timer()
        self._setup_camera_controls()
        self._setup_shortcuts()
        self._setup_deselect_shortcut()

        QtCore.QTimer.singleShot(0, self._init_engine)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Check if scene has unsaved changes
        if self._scene_dirty:
            # Show save dialog
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Unsaved Changes")
            box.setText(f"The scene '{self._scene_name}' has unsaved changes.")
            box.setInformativeText("Do you want to save your changes?")
            box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Save |
                QtWidgets.QMessageBox.StandardButton.Discard |
                QtWidgets.QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
            
            result = box.exec()
            
            if result == QtWidgets.QMessageBox.StandardButton.Save:
                self._save_scene()
                # If save failed, don't close
                if self._scene_dirty:
                    event.ignore()
                    return
            elif result == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            # Discard: just continue to close
        
        if self._window:
            self._window.close()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        self._viewport = ViewportWidget(self)
        self.setCentralWidget(self._viewport)

        self._hierarchy_dock = QtWidgets.QDockWidget("Scene", self)
        self._hierarchy_dock.setObjectName("EditorHierarchyDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self._hierarchy_dock)

        self._inspector_dock = QtWidgets.QDockWidget("Inspector", self)
        self._inspector_dock.setObjectName("EditorInspectorDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._inspector_dock)

        self._files_dock = QtWidgets.QDockWidget("Project", self)
        self._files_dock.setObjectName("EditorProjectDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._files_dock)

    def _setup_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._add_toolbar_button(toolbar, "X-", lambda: self._nudge_selected((-0.5, 0.0, 0.0)))
        self._add_toolbar_button(toolbar, "X+", lambda: self._nudge_selected((0.5, 0.0, 0.0)))
        toolbar.addSeparator()
        self._add_toolbar_button(toolbar, "Y-", lambda: self._nudge_selected((0.0, -0.5, 0.0)))
        self._add_toolbar_button(toolbar, "Y+", lambda: self._nudge_selected((0.0, 0.5, 0.0)))
        toolbar.addSeparator()
        self._add_toolbar_button(toolbar, "Z-", lambda: self._nudge_selected((0.0, 0.0, -0.5)))
        self._add_toolbar_button(toolbar, "Z+", lambda: self._nudge_selected((0.0, 0.0, 0.5)))
        
        toolbar.addSeparator()
        self._play_action = self._add_toolbar_button(toolbar, "Play", self._on_play_clicked)
        self._pause_action = self._add_toolbar_button(toolbar, "Pause", self._on_pause_clicked)
        self._stop_action = self._add_toolbar_button(toolbar, "Stop", self._on_stop_clicked)
        
        self._pause_action.setEnabled(False)
        self._stop_action.setEnabled(False)

    def _add_toolbar_button(self, toolbar: QtWidgets.QToolBar, label: str, callback) -> QtGui.QAction:
        action = QtGui.QAction(label, self)
        action.triggered.connect(callback)
        toolbar.addAction(action)
        return action

    def _setup_hierarchy_panel(self) -> None:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # Scene name label at the top
        self._scene_label = QtWidgets.QLabel("Untitled Scene")
        self._scene_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        layout.addWidget(self._scene_label)

        button_row = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add", panel)
        remove_button = QtWidgets.QPushButton("Remove", panel)
        refresh_button = QtWidgets.QPushButton("Refresh", panel)
        add_button.clicked.connect(self._show_add_menu)
        remove_button.clicked.connect(self._remove_selected)
        refresh_button.clicked.connect(self._refresh_hierarchy)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(refresh_button)
        layout.addLayout(button_row)

        self._hierarchy_tree = HierarchyTreeWidget(self, self)
        self._hierarchy_tree.setHeaderLabel("GameObjects")
        self._hierarchy_tree.itemSelectionChanged.connect(self._on_hierarchy_selection)
        self._hierarchy_tree.itemDoubleClicked.connect(self._on_hierarchy_double_click)
        self._hierarchy_tree.object_parented.connect(self._on_object_parented)
        
        layout.addWidget(self._hierarchy_tree)

        self._hierarchy_dock.setWidget(panel)

    def _on_object_parented(self, child_obj: GameObject, parent_obj: Optional[GameObject]) -> None:
        """Handle when an object is parented to another via drag-drop."""
        if not child_obj:
            return
        
        self._viewport.makeCurrent()
        
        # Store world position before parenting
        world_pos = child_obj.transform.world_position
        world_rot = child_obj.transform.world_rotation
        world_scale = child_obj.transform.world_scale
        
        if parent_obj:
            # Set parent - this will convert to local automatically
            child_obj.transform.parent = parent_obj.transform
            # Preserve world transform
            child_obj.transform.world_position = world_pos
            child_obj.transform.world_rotation = world_rot
            child_obj.transform.world_scale = world_scale
        else:
            # Unparent (make root level)
            if child_obj.transform.parent:
                child_obj.transform.parent = None
                # Restore world position
                child_obj.transform.position = world_pos
                child_obj.transform.rotation = world_rot
                child_obj.transform.scale_xyz = world_scale
        
        # Refresh the hierarchy tree
        self._refresh_hierarchy()
        
        # Defer selection to ensure widget is fully updated
        QtCore.QTimer.singleShot(0, lambda: self._select_and_expand(child_obj, parent_obj))

    def _select_and_expand(self, child_obj: GameObject, parent_obj: Optional[GameObject]) -> None:
        if parent_obj and parent_obj in self._object_items:
            self._object_items[parent_obj].setExpanded(True)
        
        self._select_object(child_obj)
        if child_obj in self._object_items:
            self._object_items[child_obj].setSelected(True)
        
        self._viewport.update()
        self._viewport.doneCurrent()
        
        # Mark scene as dirty
        self._mark_scene_dirty()

    def _setup_inspector_panel(self) -> None:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self._inspector_name = QtWidgets.QLineEdit(content)
        self._inspector_name.editingFinished.connect(self._rename_selected)
        content_layout.addWidget(QtWidgets.QLabel("Name", content))
        content_layout.addWidget(self._inspector_name)

        self._transform_group = QtWidgets.QGroupBox("Transform", content)
        form = QtWidgets.QFormLayout(self._transform_group)

        self._pos_fields = [QtWidgets.QDoubleSpinBox() for _ in range(3)]
        self._rot_fields = [QtWidgets.QDoubleSpinBox() for _ in range(3)]
        self._scale_fields = [QtWidgets.QDoubleSpinBox() for _ in range(3)]

        for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
            for f in fields:
                f.setRange(-10000, 10000)
                f.setSingleStep(0.1)
                f.setDecimals(2)
                f.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                f.valueChanged.connect(self._on_transform_changed)

        pos_row = QtWidgets.QHBoxLayout()
        for f in self._pos_fields:
            pos_row.addWidget(f)
        form.addRow("Position", pos_row)

        rot_row = QtWidgets.QHBoxLayout()
        for f in self._rot_fields:
            rot_row.addWidget(f)
        form.addRow("Rotation", rot_row)

        scale_row = QtWidgets.QHBoxLayout()
        for f in self._scale_fields:
            scale_row.addWidget(f)
        form.addRow("Scale", scale_row)

        content_layout.addWidget(self._transform_group)

        comp_header = QtWidgets.QHBoxLayout()
        comp_header.addWidget(QtWidgets.QLabel("Components"))
        add_comp_btn = QtWidgets.QPushButton("+")
        add_comp_btn.setFixedWidth(30)
        add_comp_btn.clicked.connect(self._show_add_component_menu)
        comp_header.addWidget(add_comp_btn)
        content_layout.addLayout(comp_header)

        self._components_container = QtWidgets.QWidget(content)
        self._components_layout = QtWidgets.QVBoxLayout(self._components_container)
        self._components_layout.setContentsMargins(0, 0, 0, 0)
        self._components_layout.setSpacing(6)
        self._components_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        content_layout.addWidget(self._components_container)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        self._inspector_dock.setWidget(panel)

    def _show_add_component_menu(self) -> None:
        if not self._selection.game_object:
            return

        menu = QtWidgets.QMenu(self)
        from src.engine3d.light import PointLight3D, DirectionalLight3D
        from src.physics.rigidbody import Rigidbody
        from src.physics.collider import BoxCollider, SphereCollider, CapsuleCollider
        from src.engine3d.particle import ParticleSystem

        actions = {
            "Point Light": lambda: self._add_component_to_selected(PointLight3D()),
            "Directional Light": lambda: self._add_component_to_selected(DirectionalLight3D()),
            "Box Collider": lambda: self._add_component_to_selected(BoxCollider()),
            "Sphere Collider": lambda: self._add_component_to_selected(SphereCollider()),
            "Capsule Collider": lambda: self._add_component_to_selected(CapsuleCollider()),
            "Rigidbody": lambda: self._add_component_to_selected(Rigidbody()),
            "Particle System": lambda: self._add_component_to_selected(ParticleSystem()),
        }

        for name, callback in actions.items():
            action = menu.addAction(name)
            action.triggered.connect(callback)

        # Add separator before scripts
        menu.addSeparator()
        
        # Scan for existing script files in the project
        scripts = self._find_script_files()
        if scripts:
            scripts_menu = menu.addMenu("Scripts")
            for script_path, class_name in scripts:
                action = scripts_menu.addAction(class_name)
                action.triggered.connect(lambda checked, p=script_path, c=class_name: self._add_existing_script(p, c))
        
        # Add "New Script..." option
        new_script_action = menu.addAction("New Script...")
        new_script_action.triggered.connect(self._add_script_component)

        menu.exec(QtGui.QCursor.pos())

    def _find_script_files(self) -> List[Tuple[Path, str]]:
        """
        Scan the project directory for Python files containing Script subclasses.
        
        Returns:
            List of (file_path, class_name) tuples
        """
        scripts = []
        
        # Scan all .py files in the project root
        for py_file in self.project_root.rglob("*.py"):
            # Skip files in hidden directories or __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
            
            # Skip the src directory (engine code)
            if 'src' in py_file.parts:
                continue
            
            try:
                # Read the file and look for Script subclasses
                content = py_file.read_text(encoding='utf-8')
                
                # Simple regex-like search for class definitions that inherit from Script
                import re
                pattern = r'class\s+(\w+)\s*\(\s*Script\s*\)'
                matches = re.findall(pattern, content)
                
                for class_name in matches:
                    scripts.append((py_file, class_name))
                    
            except Exception:
                # Skip files that can't be read
                continue
        
        return scripts

    def _add_existing_script(self, file_path: Path, class_name: str) -> None:
        """Load and add an existing script as a component."""
        self._load_and_add_script(file_path, class_name)

    def _add_script_component(self) -> None:
        """Open dialog to create a new script component."""
        from PySide6 import QtWidgets

        # Dialog for script name
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Script", "Enter script class name:"
        )
        if not ok or not name.strip():
            return

        script_name = name.strip()
        # Validate class name (Python identifier)
        if not script_name.isidentifier():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Name", "Script name must be a valid Python identifier."
            )
            return

        # File dialog for save location
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Script",
            str(self.project_root / f"{script_name}.py"),
            "Python Files (*.py)"
        )
        if not file_path:
            return

        file_path = Path(file_path)

        # Create the script file
        try:
            self._create_script_file(file_path, script_name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to create script file:\n{e}"
            )
            return

        # Add the script component to selected object
        self._load_and_add_script(file_path, script_name)

    def _create_script_file(self, file_path: Path, class_name: str) -> None:
        """Create a new script file with the template."""
        script_template = f'''from src.engine3d import Script, Time, InspectorField, color, vector3
from src.engine3d.transform import Transform


class {class_name}(Script):
    """
    Custom script component.
    
    Add InspectorField attributes to show them in the editor inspector.
    Example:
        speed = InspectorField(float, default=5.0, min_value=0.0, max_value=100.0)
        health = InspectorField(float, default=100.0, min_value=0.0, max_value=100.0)
        is_active = InspectorField(bool, default=True)
        player_color = InspectorField(color, default=(1.0, 0.0, 0.0))
        spawn_pos = InspectorField(vector3, default=(0.0, 0.0, 0.0))
        
    List fields - allows adding multiple values:
        scores = InspectorField(list, default=[], list_item_type=int)
        waypoints = InspectorField(list, default=[], list_item_type=float)
        
    Component reference fields - reference other components:
        player_transform = InspectorField(Transform, default=None, component_type=Transform)
        target_camera = InspectorField(Camera3D, default=None, component_type=Camera3D)
    """
    
    # Example inspector fields (uncomment to use):
    # speed = InspectorField(float, default=5.0, min_value=0.0, max_value=100.0, tooltip="Movement speed")
    # scores = InspectorField(list, default=[], list_item_type=int)
    # player_transform = InspectorField(Transform, default=None, component_type=Transform)
    
    def start(self):
        """
        Called once when the script is first initialized.
        """
        pass
    
    def update(self):
        """
        Called every frame.
        """
        pass
'''
        file_path.write_text(script_template, encoding="utf-8")

    def _load_and_add_script(self, file_path: Path, class_name: str) -> None:
        """Dynamically load the script and add it as a component."""
        import importlib.util
        import sys
        from PySide6 import QtWidgets

        try:
            # Add the project root to sys.path if not already there
            project_root = str(self.project_root)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Create a unique module name to allow reloading
            # Use the relative path from project root to create a unique identifier
            try:
                relative_path = file_path.relative_to(self.project_root)
                module_name = '.'.join(relative_path.with_suffix('').parts)
            except ValueError:
                module_name = file_path.stem
            
            # Ensure parent packages exist for dotted module names
            if "." in module_name:
                import types
                parts = module_name.split(".")
                for i in range(1, len(parts)):
                    pkg_name = ".".join(parts[:i])
                    if pkg_name not in sys.modules:
                        pkg_module = types.ModuleType(pkg_name)
                        pkg_module.__path__ = [str(self.project_root / Path(*parts[:i]))]
                        sys.modules[pkg_name] = pkg_module

            # Ensure unique module name (in case of conflicts)
            base_module_name = module_name
            counter = 1
            while module_name in sys.modules:
                # If module already exists, check if it's the same file
                existing_module = sys.modules[module_name]
                existing_path = getattr(existing_module, '__file__', None)
                if existing_path and Path(existing_path).resolve() == file_path.resolve():
                    # Same file, try to reload it
                    import importlib
                    try:
                        importlib.reload(existing_module)
                        module = existing_module
                        break
                    except Exception:
                        pass
                module_name = f"{base_module_name}_{counter}"
                counter += 1
            else:
                # Load the module fresh
                spec = importlib.util.spec_from_file_location(
                    module_name, str(file_path)
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load script from {file_path}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            # Get the class from the module
            if not hasattr(module, class_name):
                raise AttributeError(f"Script file does not contain class '{class_name}'")

            script_class = getattr(module, class_name)
            script_instance = script_class()

            # Add to selected game object
            self._add_component_to_selected(script_instance)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load script:\n{e}"
            )

    def _add_component_to_selected(self, component) -> None:
        obj = self._selection.game_object
        if not obj:
            return
        obj.add_component(component)
        self._components_dirty = True
        self._update_inspector_fields(force_components=True)
        self._viewport.update()
        self._mark_scene_dirty()
        
        # Watch the script file for changes if it's a Script component
        self._watch_script_component(component)

    def _remove_component(self, component) -> None:
        """Remove a component from the selected game object."""
        obj = self._selection.game_object
        if not obj:
            return
        
        # Don't allow removing Transform
        from src.engine3d.transform import Transform
        if isinstance(component, Transform):
            QtWidgets.QMessageBox.warning(self, "Cannot Remove", "Cannot remove Transform component.")
            return
        
        # Check if component belongs to the selected object
        if component not in obj.components:
            return
        
        # Remove the component
        obj.components.remove(component)
        component.game_object = None
        
        # Refresh the inspector
        self._components_dirty = True
        self._update_inspector_fields(force_components=True)
        self._viewport.update()
        self._mark_scene_dirty()

    def _watch_script_component(self, component) -> None:
        """Add a script component's source file to the file watcher."""
        from src.engine3d.component import Script
        
        if not isinstance(component, Script):
            return
        
        # Get the source file of the component's class
        import inspect
        try:
            source_file = inspect.getfile(type(component))
            if source_file and source_file.endswith('.py'):
                # Check if it's in the project directory (not engine code)
                source_path = Path(source_file).resolve()
                try:
                    source_path.relative_to(self.project_root)
                    # It's a project file, watch it
                    if source_file not in self._watched_script_files:
                        self._file_watcher.addPath(source_file)
                        self._watched_script_files[source_file] = source_path.stat().st_mtime
                except ValueError:
                    # Not in project directory, skip
                    pass
        except (TypeError, OSError):
            # Built-in or compiled module, skip
            pass

    def _on_script_file_changed(self, path: str) -> None:
        """Handle when a watched script file changes."""
        import time
        
        # Check if the file still exists
        if not Path(path).exists():
            return
        
        # Get current modification time
        try:
            current_mtime = Path(path).stat().st_mtime
        except OSError:
            return
        
        # Check if this is a real change (not just a save trigger)
        last_mtime = self._watched_script_files.get(path, 0)
        if current_mtime <= last_mtime:
            return
        
        self._watched_script_files[path] = current_mtime
        
        # Re-add the file to the watcher (some editors delete and recreate)
        if path not in self._file_watcher.files():
            self._file_watcher.addPath(path)
        
        # Debounce the reload to handle editors that make multiple saves
        if not self._debounce_timer.isActive():
            self._debounce_timer.start(500)  # 500ms debounce

    def _reload_script_components(self) -> None:
        """Reload all script components in the scene when code changes."""
        import importlib
        import sys
        import inspect
        
        # Don't reload during play mode
        if self._playing:
            return
        
        # Collect all script components and their source files
        scripts_by_file: Dict[str, List[tuple]] = {}  # file -> [(component, gameobject), ...]
        
        for obj in self._scene.objects:
            for comp in obj.components:
                from src.engine3d.component import Script
                if isinstance(comp, Script):
                    try:
                        source_file = inspect.getfile(type(comp))
                        if source_file in self._watched_script_files:
                            if source_file not in scripts_by_file:
                                scripts_by_file[source_file] = []
                            scripts_by_file[source_file].append((comp, obj))
                    except (TypeError, OSError):
                        continue
        
        if not scripts_by_file:
            return
        
        # Reload each affected module
        reloaded_modules = set()
        for source_file, components in scripts_by_file.items():
            try:
                # Find the module for this source file
                module_name = None
                for name, module in sys.modules.items():
                    if hasattr(module, '__file__') and module.__file__ == source_file:
                        module_name = name
                        break
                
                if module_name and module_name not in reloaded_modules:
                    # Reload the module
                    importlib.reload(sys.modules[module_name])
                    reloaded_modules.add(module_name)
                    
                    # Get the new class from the reloaded module
                    old_class = type(components[0][0])
                    new_class = getattr(sys.modules[module_name], old_class.__name__, None)
                    
                    if new_class and new_class is not old_class:
                        # Update all instances of this class
                        for old_comp, game_obj in components:
                            # Store the old values
                            old_values = {}
                            for name, info in old_comp.get_inspector_fields():
                                old_values[name] = old_comp.get_inspector_field_value(name)
                            
                            # Create new instance
                            new_comp = new_class()
                            
                            # Copy over the game_object reference
                            new_comp.game_object = game_obj
                            
                            # Restore old values
                            for name, value in old_values.items():
                                new_comp.set_inspector_field_value(name, value)
                            
                            # Replace the component in the game object
                            idx = game_obj.components.index(old_comp)
                            game_obj.components[idx] = new_comp
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error reloading script {source_file}: {e}")
        
        # Mark components as dirty to refresh the inspector
        self._components_dirty = True
        self._update_inspector_fields(force_components=True)
        self._viewport.update()

    def _setup_files_panel(self) -> None:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        self._file_model = QtWidgets.QFileSystemModel(panel)
        self._file_model.setRootPath(str(self.project_root))
        self._file_model.setFilter(QtCore.QDir.Filter.AllEntries | QtCore.QDir.Filter.NoDotAndDotDot)

        self._file_view = QtWidgets.QTreeView(panel)
        self._file_view.setModel(self._file_model)
        self._file_view.setRootIndex(self._file_model.index(str(self.project_root)))
        self._file_view.setColumnWidth(0, 280)
        self._file_view.setDragEnabled(True)
        self._file_view.doubleClicked.connect(self._on_file_double_clicked)
        layout.addWidget(self._file_view)

        self._files_dock.setWidget(panel)

        # Connect viewport drop signal
        self._viewport.file_dropped.connect(self._on_file_dropped)

    def _on_file_double_clicked(self, index: QtCore.QModelIndex) -> None:
        path = self._file_model.filePath(index)
        self._add_3d_object_from_path(path)

    def _on_file_dropped(self, path: str) -> None:
        if not path:
            # Drop from tree view
            index = self._file_view.currentIndex()
            if index.isValid():
                path = self._file_model.filePath(index)
        
        if path:
            self._add_3d_object_from_path(path)

    def _add_3d_object_from_path(self, path: str) -> None:
        ext = Path(path).suffix.lower()
        # Common 3D file extensions supported by trimesh
        if ext in {'.obj', '.gltf', '.glb', '.stl', '.ply', '.off'}:
            try:
                self._viewport.makeCurrent()
                obj3d = Object3D(path)
                go = GameObject(Path(path).stem)
                go.add_component(obj3d)
                
                # Position in front of camera (at target)
                go.transform.position = tuple(self._camera_control['target'])
                
                self._scene.add_object(go)
                self._refresh_hierarchy()
                self._select_object(go)
                self._viewport.update()
                self._viewport.doneCurrent()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load 3D object:\n{e}")

    def _on_play_clicked(self) -> None:
        """Run the current scene as a game in the viewport."""
        if self._playing:
            return

        try:
            # Store original scene state
            self._original_scene_data = self._scene._to_scene_dict()
            
            # Switch to game camera
            if self._window:
                self._window.active_camera_override = None
                self._window.editor_show_axis = False
            
            # Initialize all scripts
            for obj in self._scene.objects:
                obj.start_scripts()
            
            self._playing = True
            self._paused = False
            
            self._play_action.setEnabled(False)
            self._pause_action.setEnabled(True)
            self._stop_action.setEnabled(True)
            self._pause_action.setText("Pause")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_text = traceback.format_exc()
            
            # Show error dialog with details
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("Play Mode Error")
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg_box.setText("An error occurred while starting play mode.")
            msg_box.setInformativeText(f"Error: {error_msg}\n\nPlay mode could not be started.")
            msg_box.setDetailedText(traceback_text)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg_box.exec()

    def _on_pause_clicked(self) -> None:
        """Toggle pause state."""
        if not self._playing:
            return
            
        self._paused = not self._paused
        self._pause_action.setText("Resume" if self._paused else "Pause")

    def _on_stop_clicked(self) -> None:
        """Stop play mode and restore scene state."""
        if not self._playing:
            return

        try:
            self._playing = False
            self._paused = False
            
            # Restore editor camera
            if self._window:
                self._window.active_camera_override = self._editor_camera
                self._window.editor_show_axis = True
            
            # Restore scene state
            if self._original_scene_data:
                # We need to be careful with the viewport context when restoring
                self._viewport.makeCurrent()
                # Clear current scene's GPU resources
                self._window.clear_objects()
                
                # Re-create scene from data
                new_scene = EditorScene._from_scene_dict(self._original_scene_data)
                self._scene = new_scene
                self._window.show_scene(self._scene)
                
                self._refresh_hierarchy()
                self._select_object(None)
                self._viewport.update()
                self._viewport.doneCurrent()
            
            self._play_action.setEnabled(True)
            self._pause_action.setEnabled(False)
            self._stop_action.setEnabled(False)
            self._pause_action.setText("Pause")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to stop play mode:\n{e}")

    def _on_play_mode_error(self, error_msg: str, traceback_text: str) -> None:
        """
        Handle an error that occurred during play mode.
        Stops play mode and shows an error dialog.
        """
        # Stop play mode first
        self._on_stop_clicked()
        
        # Show error dialog with details
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Play Mode Error")
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setText("An error occurred during play mode.")
        msg_box.setInformativeText(f"Error: {error_msg}\n\nPlay mode has been stopped.")
        msg_box.setDetailedText(traceback_text)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def _setup_timer(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick_engine)

    def _mark_components_dirty(self) -> None:
        self._components_dirty = True

    def _clear_component_fields(self) -> None:
        for widget in self._component_fields:
            widget.setParent(None)
            widget.deleteLater()
        self._component_fields.clear()

    def _apply_spinbox(self, spinbox: QtWidgets.QDoubleSpinBox, value: float) -> None:
        if not spinbox.hasFocus():
            spinbox.setValue(value)

    def _apply_slider(self, slider: QtWidgets.QSlider, value: int) -> None:
        if not slider.hasFocus():
            slider.setValue(value)

    def _make_spinbox(self, minimum: float, maximum: float, step: float = 0.1, decimals: int = 2) -> QtWidgets.QDoubleSpinBox:
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        return spinbox

    def _make_vector_row(self, values: Iterable[float], on_changed, minimum: float = -10000.0, maximum: float = 10000.0,
                         step: float = 0.1, decimals: int = 2) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = []
        for value in values:
            spin = self._make_spinbox(minimum, maximum, step, decimals)
            spin.setValue(value)
            spin.valueChanged.connect(on_changed)
            layout.addWidget(spin)
            fields.append(spin)
        widget._vector_fields = fields
        return widget

    def _make_color_slider(self, channel_name: str, initial: int, on_changed) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(channel_name)
        label.setFixedWidth(12)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(initial)
        slider.valueChanged.connect(on_changed)
        value_label = QtWidgets.QLabel(str(initial))
        value_label.setFixedWidth(32)
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        widget._color_slider = slider
        widget._value_label = value_label
        return widget

    def _set_component_box(self, component_box: QtWidgets.QGroupBox, component_name: str) -> None:
        component_box.setTitle(component_name)
        component_box.setProperty("component_name", component_name)

    def _ensure_component_box(self, component_box: QtWidgets.QGroupBox) -> None:
        if component_box in self._component_fields:
            return
        self._component_fields.append(component_box)
        self._components_layout.addWidget(component_box)

    def _update_component_box_title(self, component_box: QtWidgets.QGroupBox, name: str) -> None:
        if component_box.title() != name:
            component_box.setTitle(name)

    def _init_engine(self) -> None:
        if self._window:
            return

        self._viewport.makeCurrent()
        dpr = self._viewport.devicePixelRatio()

        self._window = Window3D(
            width=int(max(1, self._viewport.width() * dpr)),
            height=int(max(1, self._viewport.height() * dpr)),
            title="Engine3D Editor Viewport",
            resizable=True,
            use_pygame_window=False,
            use_pygame_events=False,
        )
        self._window.show_editor_overlays = True
        self._window.editor_show_camera = True
        self._window.active_camera_override = self._editor_camera
        
        # Initialize scene management
        self._init_scene_file()
        
        self._window.show_scene(self._scene)

        self._viewport.resized.connect(self._on_viewport_resized)

        # Initialize camera using spherical coordinates
        self._update_camera_position()

        self._refresh_hierarchy()
        self._select_object(None)

        if not self._scene.objects:
            self._update_inspector_fields()

        self._viewport.render_callback = self._render_frame
        self._timer.start()

    def _init_scene_file(self) -> None:
        """Initialize the Scenes folder and main scene file."""
        scenes_dir = self.project_root / "Scenes"
        main_scene_path = scenes_dir / "main.scene"
        
        # Create Scenes directory if it doesn't exist
        if not scenes_dir.exists():
            scenes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create main.scene
        if main_scene_path.exists():
            self._load_scene(main_scene_path)
        else:
            # Create empty main scene file
            self._current_scene_path = main_scene_path
            self._scene_name = "main"
            self._scene.editor_label = "main"
            self._save_scene()  # Save the empty scene
            self._scene_dirty = False  # Not dirty since we just created it
        
        self._update_scene_label()

    def _load_scene(self, path: Path) -> None:
        """Load a scene from a file."""
        try:
            self._viewport.makeCurrent()
            
            # Clear current scene
            if self._window:
                self._window.clear_objects()
            
            # Load the scene
            from src.editor.scene import EditorScene
            self._scene = EditorScene.load(str(path))
            self._current_scene_path = path
            self._scene_name = path.stem
            self._scene.editor_label = self._scene_name
            self._scene_dirty = False
            
            # Show the loaded scene
            if self._window:
                self._window.show_scene(self._scene)
            
            self._refresh_hierarchy()
            self._select_object(None)
            self._viewport.update()
            self._viewport.doneCurrent()
            
            self._update_scene_label()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load scene:\n{e}")

    def _save_scene(self) -> None:
        """Save the current scene to its file."""
        if self._current_scene_path is None:
            self._save_scene_as()
            return
        
        try:
            self._scene.save(str(self._current_scene_path))
            self._scene_dirty = False
            self._update_scene_label()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save scene:\n{e}")

    def _save_scene_as(self) -> None:
        """Save the current scene to a new file."""
        scenes_dir = self.project_root / "Scenes"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scene",
            str(scenes_dir / "new_scene.scene"),
            "Scene Files (*.scene)"
        )
        
        if file_path:
            path = Path(file_path)
            try:
                self._scene.save(str(path))
                self._current_scene_path = path
                self._scene_name = path.stem
                self._scene.editor_label = self._scene_name
                self._scene_dirty = False
                self._update_scene_label()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save scene:\n{e}")

    def _mark_scene_dirty(self) -> None:
        """Mark the scene as having unsaved changes."""
        if not self._scene_dirty:
            self._scene_dirty = True
            self._update_scene_label()

    def _update_scene_label(self) -> None:
        """Update the scene name label in the hierarchy panel."""
        if hasattr(self, '_scene_label'):
            display_name = self._scene_name
            if self._scene_dirty:
                display_name = f"*{self._scene_name}"
            self._scene_label.setText(display_name)

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        # Ctrl+S to save
        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._save_scene)
        
        # Ctrl+Shift+S to save as
        save_as_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self)
        save_as_shortcut.activated.connect(self._save_scene_as)
        
        # Ctrl+O to open scene
        open_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self._open_scene_dialog)

    def _setup_deselect_shortcut(self) -> None:
        esc_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        esc_shortcut.activated.connect(self._deselect_all)

    def _deselect_all(self) -> None:
        self._hierarchy_tree.clearSelection()
        self._select_object(None)

    def _open_scene_dialog(self) -> None:
        """Open a dialog to select a scene to load."""
        scenes_dir = self.project_root / "Scenes"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Scene",
            str(scenes_dir),
            "Scene Files (*.scene)"
        )
        
        if file_path:
            self._load_scene(Path(file_path))

    def _render_frame(self) -> None:
        """Called by ViewportWidget.paintGL() to render the frame."""
        if not self._window:
            return
        
        try:
            # Update moderngl framebuffer wrapper if ID changed (e.g. after resize)
            fbo_id = self._viewport.defaultFramebufferObject()
            if not hasattr(self, '_last_fbo_id') or self._last_fbo_id != fbo_id:
                self._last_fbo_id = fbo_id
                self._window._screen_fbo = self._window._ctx.detect_framebuffer()
            
            # Ensure moderngl knows about it
            if getattr(self._window, '_screen_fbo', None):
                self._window._screen_fbo.use()
                
            simulate = self._playing and not self._paused
            if not self._window.tick(simulate=simulate):
                self._timer.stop()
        except Exception as e:
            # Handle errors during play mode
            if self._playing:
                import traceback
                error_msg = str(e)
                traceback_text = traceback.format_exc()
                self.play_mode_error.emit(error_msg, traceback_text)
            else:
                # Re-raise if not in play mode
                raise

    def _tick_engine(self) -> None:
        """Called by timer to request a redraw and update UI state."""
        if not self._window:
            return
        self._viewport.update()  # Triggers paintGL
        self._update_inspector_fields()

    def _on_viewport_resized(self, width: int, height: int) -> None:
        if not self._window:
            return
        self._viewport.makeCurrent()
        try:
            self._window.on_resize(width, height)
        finally:
            self._viewport.doneCurrent()

    def _setup_camera_controls(self) -> None:
        """Setup Unity-style camera controls (orbit, pan, zoom)."""
        self._viewport.mouse_pressed.connect(self._on_mouse_pressed)
        self._viewport.mouse_released.connect(self._on_mouse_released)
        self._viewport.mouse_moved.connect(self._on_mouse_moved)
        self._viewport.wheel_scrolled.connect(self._on_wheel_scrolled)
        self._viewport.key_pressed.connect(self._on_key_pressed)
        self._viewport.key_released.connect(self._on_key_released)

    def _on_mouse_pressed(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse button press for camera control."""
        if self._playing and not self._paused:
            # Forward to engine
            button = 0
            if event.button() == QtCore.Qt.MouseButton.LeftButton: button = 1
            elif event.button() == QtCore.Qt.MouseButton.MiddleButton: button = 2
            elif event.button() == QtCore.Qt.MouseButton.RightButton: button = 3
            if button > 0:
                self._window._mouse_buttons.add(button)
                self._scene.on_mouse_press(event.pos().x(), event.pos().y(), button, 0)
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            # Right-click: Orbit
            self._camera_control['orbiting'] = True
            self._camera_control['last_mouse_pos'] = (event.pos().x(), event.pos().y())
            self._viewport.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            # Middle-click: Pan
            self._camera_control['panning'] = True
            self._camera_control['last_mouse_pos'] = (event.pos().x(), event.pos().y())
            self._viewport.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def _on_mouse_released(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse button release for camera control."""
        if self._playing and not self._paused:
            button = 0
            if event.button() == QtCore.Qt.MouseButton.LeftButton: button = 1
            elif event.button() == QtCore.Qt.MouseButton.MiddleButton: button = 2
            elif event.button() == QtCore.Qt.MouseButton.RightButton: button = 3
            if button > 0:
                self._window._mouse_buttons.discard(button)
                self._scene.on_mouse_release(event.pos().x(), event.pos().y(), button, 0)
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._camera_control['orbiting'] = False
            self._viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._camera_control['panning'] = False
            self._viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self._camera_control['last_mouse_pos'] = None

    def _on_mouse_moved(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement for camera control."""
        if not self._window:
            return

        current_pos = (event.pos().x(), event.pos().y())
        
        if self._playing and not self._paused:
            dx = 0
            dy = 0
            if self._camera_control['last_mouse_pos']:
                dx = current_pos[0] - self._camera_control['last_mouse_pos'][0]
                dy = current_pos[1] - self._camera_control['last_mouse_pos'][1]
            self._window._mouse_position = current_pos
            self._scene.on_mouse_motion(current_pos[0], current_pos[1], dx, dy)
            self._camera_control['last_mouse_pos'] = current_pos
            return

        last_pos = self._camera_control['last_mouse_pos']
        if last_pos is None:
            return

        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]

        if self._camera_control['orbiting']:
            # Orbit around target
            sensitivity = 0.5
            self._camera_control['azimuth'] -= dx * sensitivity
            self._camera_control['elevation'] += dy * sensitivity
            # Clamp elevation to avoid flipping
            self._camera_control['elevation'] = np.clip(self._camera_control['elevation'], -89.0, 89.0)
            self._update_camera_position()
            
        elif self._camera_control['panning']:
            # Pan the target point
            sensitivity = 0.01 * self._camera_control['distance']
            
            # Calculate right and up vectors based on current camera orientation
            azimuth_rad = np.radians(self._camera_control['azimuth'])
            elevation_rad = np.radians(self._camera_control['elevation'])
            
            # Forward vector (from camera to target)
            forward = np.array([
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
                np.cos(elevation_rad) * np.cos(azimuth_rad)
            ], dtype=np.float32)
            forward = -forward  # Camera looks at target, so forward is opposite
            
            # Right vector
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
            if right_norm > 0.001:
                right = right / right_norm
            else:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
            # Up vector
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Pan target
            pan_x = -dx * sensitivity
            pan_y = dy * sensitivity
            
            self._camera_control['target'] += right * pan_x + up * pan_y
            self._update_camera_position()

        self._camera_control['last_mouse_pos'] = current_pos

    def _on_key_pressed(self, event: QtGui.QKeyEvent) -> None:
        if not self._playing or self._paused:
            return
        
        key = self._map_qt_key_to_pygame(event.key())
        if key:
            self._window._keys_pressed.add(key)
            self._scene.on_key_press(key, 0)

    def _on_key_released(self, event: QtGui.QKeyEvent) -> None:
        if not self._playing or self._paused:
            return
            
        key = self._map_qt_key_to_pygame(event.key())
        if key:
            self._window._keys_pressed.discard(key)
            self._scene.on_key_release(key, 0)

    def _map_qt_key_to_pygame(self, qt_key: int) -> Optional[int]:
        import pygame
        # Basic mapping for common keys
        mapping = {
            QtCore.Qt.Key.Key_W: pygame.K_w,
            QtCore.Qt.Key.Key_A: pygame.K_a,
            QtCore.Qt.Key.Key_S: pygame.K_s,
            QtCore.Qt.Key.Key_D: pygame.K_d,
            QtCore.Qt.Key.Key_Q: pygame.K_q,
            QtCore.Qt.Key.Key_E: pygame.K_e,
            QtCore.Qt.Key.Key_Space: pygame.K_SPACE,
            QtCore.Qt.Key.Key_Shift: pygame.K_LSHIFT,
            QtCore.Qt.Key.Key_Control: pygame.K_LCTRL,
            QtCore.Qt.Key.Key_Alt: pygame.K_LALT,
            QtCore.Qt.Key.Key_Escape: pygame.K_ESCAPE,
            QtCore.Qt.Key.Key_Up: pygame.K_UP,
            QtCore.Qt.Key.Key_Down: pygame.K_DOWN,
            QtCore.Qt.Key.Key_Left: pygame.K_LEFT,
            QtCore.Qt.Key.Key_Right: pygame.K_RIGHT,
        }
        # For letters, we can also try direct mapping if not in dict
        if qt_key >= QtCore.Qt.Key.Key_A and qt_key <= QtCore.Qt.Key.Key_Z:
            return pygame.K_a + (qt_key - QtCore.Qt.Key.Key_A)
        
        return mapping.get(qt_key)

    def _on_wheel_scrolled(self, event: QtGui.QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if not self._window:
            return

        # Get scroll delta
        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        
        # Apply zoom
        self._camera_control['distance'] *= zoom_factor
        # Clamp distance
        self._camera_control['distance'] = np.clip(self._camera_control['distance'], 0.1, 1000.0)
        
        self._update_camera_position()

    def _update_camera_position(self) -> None:
        """Update camera position based on spherical coordinates."""
        if not self._window:
            return

        azimuth_rad = np.radians(self._camera_control['azimuth'])
        elevation_rad = np.radians(self._camera_control['elevation'])
        distance = self._camera_control['distance']
        target = self._camera_control['target']

        # Calculate camera position on sphere around target
        # Azimuth: rotation around Y axis (0 = looking along -Z)
        # Elevation: angle from horizontal plane
        cam_offset = np.array([
            distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
            distance * np.sin(elevation_rad),
            distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        ], dtype=np.float32)

        camera_pos = target + cam_offset
        
        # Update editor camera
        # Create a dummy GameObject for the editor camera if it doesn't have one
        # so that look_at works correctly (it needs a transform)
        if not self._editor_camera.game_object:
            from src.engine3d.gameobject import GameObject
            cam_go = GameObject("Editor Camera")
            cam_go.add_component(self._editor_camera)
            
        self._editor_camera.game_object.transform.position = tuple(camera_pos)
        self._editor_camera.game_object.transform.look_at(tuple(target))

        if self._selection.game_object and self._selection.game_object.name == "Editor Camera":
            self._update_inspector_fields()

        self._viewport.update()

    def _refresh_hierarchy(self) -> None:
        self._hierarchy_tree.clear()
        self._object_items.clear()
        
        # Build hierarchy based on transform parent-child relationships
        # First, collect all non-auto objects
        all_objects = [obj for obj in self._scene.objects if obj.name not in self._scene_auto_objects]
        
        # Track which objects have been added
        added = set()
        
        def add_object_to_tree(obj: GameObject, parent_item=None):
            """Recursively add object and its children to the tree."""
            if obj in added:
                return
            added.add(obj)
            
            item = QtWidgets.QTreeWidgetItem([obj.name])
            self._object_items[obj] = item
            
            if parent_item:
                parent_item.addChild(item)
            else:
                self._hierarchy_tree.addTopLevelItem(item)
            
            # Add children (objects whose transform parent is this object's transform)
            for child_obj in all_objects:
                if child_obj not in added:
                    if child_obj.transform.parent is obj.transform:
                        add_object_to_tree(child_obj, item)
        
        # First pass: add root objects (no parent or parent not in scene)
        for obj in all_objects:
            if obj.transform.parent is None:
                add_object_to_tree(obj)
        
        # Second pass: add remaining objects (those with parents not in the hierarchy)
        for obj in all_objects:
            if obj not in added:
                add_object_to_tree(obj)
        
        # Set up expand/collapse icon indicators
        self._hierarchy_tree.setRootIsDecorated(True)
        self._hierarchy_tree.setItemsExpandable(True)
        for obj, item in self._object_items.items():
            if self._get_object_children(obj, all_objects):
                item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            else:
                item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator)

    def _get_object_children(self, obj: GameObject, all_objects: List[GameObject]) -> List[GameObject]:
        return [child for child in all_objects if child.transform.parent is obj.transform]

    def _show_add_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        
        # Determine parent for the new object
        # If an object is selected, the new object will be its child
        parent_obj = self._selection.game_object
        
        empty_action = menu.addAction("Empty GameObject")
        cube_action = menu.addAction("Cube")
        sphere_action = menu.addAction("Sphere")
        plane_action = menu.addAction("Plane")
        camera_action = menu.addAction("Camera")
        
        action = menu.exec(QtGui.QCursor.pos())
        if not action:
            return

        new_obj = None
        name = ""

        if action == empty_action:
            new_obj = GameObject()
            name = "GameObject"
        elif action == cube_action:
            new_obj = create_cube(1.0)
            name = "Cube"
        elif action == sphere_action:
            new_obj = create_sphere(0.75)
            name = "Sphere"
        elif action == plane_action:
            new_obj = create_plane(5.0, 5.0)
            name = "Plane"
        elif action == camera_action:
            from src.engine3d.camera import Camera3D
            new_obj = GameObject("Camera")
            new_obj.add_component(Camera3D())
            name = "Camera"

        if new_obj:
            if parent_obj:
                new_obj.transform.parent = parent_obj.transform
            self._add_object(new_obj, name)

    def _add_object(self, obj: GameObject, name: str) -> None:
        self._viewport.makeCurrent()
        obj.name = name
        self._scene.add_object(obj)
        self._refresh_hierarchy()
        
        # Defer selection to ensure widget is fully updated
        parent_obj = obj.transform.parent.game_object if obj.transform.parent else None
        QtCore.QTimer.singleShot(0, lambda: self._select_and_expand(obj, parent_obj))
        
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()

    def _remove_selected(self) -> None:
        if not self._selection.game_object:
            return
        self._viewport.makeCurrent()
        obj = self._selection.game_object
        self._scene.remove_object(obj)
        self._selection.game_object = None
        self._refresh_hierarchy()
        self._update_inspector_fields(force_components=True)
        if self._window:
            self._window.editor_selected_object = None
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()

    def _on_hierarchy_selection(self) -> None:
        items = self._hierarchy_tree.selectedItems()
        if not items:
            self._select_object(None)
            return
        selected_item = items[0]
        for obj, item in self._object_items.items():
            if item is selected_item:
                self._select_object(obj)
                return
        # If not found directly, it might be a child item (shouldn't happen with our dict, but just in case)
        self._select_object(None)

    def _on_hierarchy_double_click(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        for obj, it in self._object_items.items():
            if it is item:
                self._focus_on_object(obj)
                break

    def _focus_on_object(self, obj: GameObject) -> None:
        if not self._window:
            return
        
        # Update the camera control target to the object's position
        target = obj.transform.world_position
        self._camera_control['target'] = np.array(target, dtype=np.float32)
        
        # Keep current distance but update position
        self._update_camera_position()

    def _select_object(self, obj: Optional[GameObject]) -> None:
        self._selection.game_object = obj
        if self._window:
            self._window.editor_selected_object = obj

        self._components_dirty = True

        # Block signals to avoid feedback loop while updating UI
        self._set_inspector_signals_blocked(True)
        if obj and obj in self._object_items:
            self._hierarchy_tree.setCurrentItem(self._object_items[obj])
        self._update_inspector_fields(force_components=True)
        self._set_inspector_signals_blocked(False)
        self._components_dirty = False

    def _set_inspector_signals_blocked(self, blocked: bool) -> None:
        for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
            for f in fields:
                f.blockSignals(blocked)
        self._inspector_name.blockSignals(blocked)
        for widget in self._component_fields:
            widget.blockSignals(blocked)
            for child in widget.findChildren(QtWidgets.QWidget):
                child.blockSignals(blocked)

    def _rename_selected(self) -> None:
        obj = self._selection.game_object
        if not obj:
            return
        name = self._inspector_name.text().strip()
        if not name:
            return
        obj.name = name
        if obj.name in self._scene_auto_objects:
            return
        if obj in self._object_items:
            self._object_items[obj].setText(0, name)
        self._viewport.update()

    def _on_transform_changed(self) -> None:
        obj = self._selection.game_object
        if not obj:
            return

        pos = [f.value() for f in self._pos_fields]
        rot = [f.value() for f in self._rot_fields]
        scale = [f.value() for f in self._scale_fields]

        obj.transform.position = pos
        obj.transform.rotation = rot
        obj.transform.scale_xyz = scale
        if self._window:
            self._window.editor_selected_object = obj
        self._viewport.update()
        self._mark_scene_dirty()

    def _nudge_selected(self, delta) -> None:
        obj = self._selection.game_object
        if not obj:
            return
        obj.transform.move(*delta)
        if self._window:
            self._window.editor_selected_object = obj
        self._viewport.update()
        self._set_inspector_signals_blocked(True)
        self._update_inspector_fields()
        self._set_inspector_signals_blocked(False)

    def _update_inspector_fields(self, force_components: bool = False) -> None:
        obj = self._selection.game_object
        if not obj:
            self._inspector_name.setText("")
            self._inspector_name.setEnabled(False)
            for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
                for f in fields:
                    f.setValue(0.0)
                    f.setEnabled(False)
            self._transform_group.setVisible(False)
            self._clear_component_fields()
            self._components_dirty = True
            return
        
        self._inspector_name.setEnabled(True)
        self._transform_group.setVisible(True)
        for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
            for f in fields:
                f.setEnabled(True)

        if force_components:
            self._components_dirty = True

        if not self._inspector_name.hasFocus():
            self._inspector_name.setText(obj.name)

        pos = obj.transform.position
        rot = obj.transform.rotation
        scale = obj.transform.scale_xyz

        fields_data = [
            (self._pos_fields, pos),
            (self._rot_fields, rot),
            (self._scale_fields, scale),
        ]

        for fields, values in fields_data:
            for i, f in enumerate(fields):
                if not f.hasFocus():
                    f.setValue(values[i])

        if force_components or self._components_dirty:
            self._build_component_fields(obj)
        else:
            self._refresh_component_fields(obj)

    def _build_component_fields(self, obj: GameObject) -> None:
        from src.engine3d.light import Light3D, DirectionalLight3D, PointLight3D
        from src.physics.collider import Collider, BoxCollider, SphereCollider, CapsuleCollider
        from src.engine3d.object3d import Object3D
        from src.physics.rigidbody import Rigidbody
        self._clear_component_fields()

        for comp in obj.components:
            if comp is obj.transform:
                continue
            
            # Get inspector fields from the component
            inspector_fields = comp.get_inspector_fields()
            
            if inspector_fields:
                # Build fields dynamically using InspectorField metadata
                box = self._create_inspector_fields_for_component(comp, inspector_fields)
            elif isinstance(comp, Light3D):
                # Fallback for old-style components (shouldn't happen if properly updated)
                if isinstance(comp, DirectionalLight3D):
                    box = self._create_directional_light_fields(comp)
                elif isinstance(comp, PointLight3D):
                    box = self._create_point_light_fields(comp)
                else:
                    box = self._create_light_fields(comp)
            elif isinstance(comp, Collider):
                if isinstance(comp, BoxCollider):
                    box = self._create_box_collider_fields(comp)
                elif isinstance(comp, SphereCollider):
                    box = self._create_sphere_collider_fields(comp)
                elif isinstance(comp, CapsuleCollider):
                    box = self._create_capsule_collider_fields(comp)
                else:
                    box = self._create_collider_fields(comp)
            elif isinstance(comp, Object3D):
                box = self._create_object3d_fields(comp)
            elif isinstance(comp, Rigidbody):
                box = self._create_rigidbody_fields(comp)
            else:
                box = self._create_component_summary(comp)
            box._component_ref = comp
            self._ensure_component_box(box)

        self._components_dirty = False

    def _create_inspector_fields_for_component(self, comp, inspector_fields: List) -> QtWidgets.QGroupBox:
        """
        Create inspector UI for a component based on its InspectorField definitions.
        
        Args:
            comp: The component instance
            inspector_fields: List of (name, InspectorFieldInfo) tuples
            
        Returns:
            A QGroupBox containing the inspector fields
        """
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a form layout for the fields
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        
        # Store field widgets for updating
        field_widgets = {}
        
        for field_name, field_info in inspector_fields:
            widget = self._create_widget_for_field(comp, field_name, field_info)
            if widget:
                form_layout.addRow(self._format_field_label(field_name), widget)
                field_widgets[field_name] = widget
        
        main_layout.addLayout(form_layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        box._inspector_field_widgets = field_widgets
        return box

    def _format_field_label(self, field_name: str) -> str:
        """Format a field name as a human-readable label."""
        # Convert snake_case to Title Case
        words = field_name.replace('_', ' ').split()
        return ' '.join(word.capitalize() for word in words)

    def _create_widget_for_field(self, comp, field_name: str, field_info) -> QtWidgets.QWidget:
        """
        Create the appropriate widget for an inspector field based on its type.
        
        Args:
            comp: The component instance
            field_name: The name of the field
            field_info: InspectorFieldInfo instance
            
        Returns:
            A QWidget for editing the field value
        """
        current_value = comp.get_inspector_field_value(field_name)
        
        if field_info.field_type == InspectorFieldType.FLOAT:
            return self._create_float_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.INT:
            return self._create_int_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.BOOL:
            return self._create_bool_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.STRING:
            return self._create_string_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.COLOR:
            return self._create_color_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.VECTOR3:
            return self._create_vector3_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.ENUM:
            return self._create_enum_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.LIST:
            return self._create_list_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.COMPONENT_REF:
            return self._create_component_ref_field(comp, field_name, field_info, current_value)
        else:
            # Fallback: just show a label
            label = QtWidgets.QLabel(str(current_value))
            return label

    def _create_float_field(self, comp, field_name: str, field_info, current_value: float) -> QtWidgets.QDoubleSpinBox:
        """Create a spinbox for a float field."""
        spinbox = QtWidgets.QDoubleSpinBox()
        min_val = field_info.min_value if field_info.min_value is not None else -10000.0
        max_val = field_info.max_value if field_info.max_value is not None else 10000.0
        step = field_info.step if field_info.step is not None else 0.1
        decimals = field_info.decimals if field_info.decimals is not None else 2
        
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        spinbox.setValue(float(current_value) if current_value is not None else field_info.default_value)
        
        if field_info.tooltip:
            spinbox.setToolTip(field_info.tooltip)
        
        spinbox.valueChanged.connect(lambda val, c=comp, fn=field_name: self._on_inspector_field_changed(c, fn, val))
        return spinbox

    def _create_int_field(self, comp, field_name: str, field_info, current_value: int) -> QtWidgets.QSpinBox:
        """Create a spinbox for an int field."""
        spinbox = QtWidgets.QSpinBox()
        min_val = int(field_info.min_value) if field_info.min_value is not None else -10000
        max_val = int(field_info.max_value) if field_info.max_value is not None else 10000
        step = int(field_info.step) if field_info.step is not None else 1
        
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        spinbox.setValue(int(current_value) if current_value is not None else field_info.default_value)
        
        if field_info.tooltip:
            spinbox.setToolTip(field_info.tooltip)
        
        spinbox.valueChanged.connect(lambda val, c=comp, fn=field_name: self._on_inspector_field_changed(c, fn, val))
        return spinbox

    def _create_bool_field(self, comp, field_name: str, field_info, current_value: bool) -> QtWidgets.QCheckBox:
        """Create a checkbox for a bool field."""
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(bool(current_value) if current_value is not None else field_info.default_value)
        
        if field_info.tooltip:
            checkbox.setToolTip(field_info.tooltip)
        
        checkbox.toggled.connect(lambda val, c=comp, fn=field_name: self._on_inspector_field_changed(c, fn, val))
        return checkbox

    def _create_string_field(self, comp, field_name: str, field_info, current_value: str) -> QtWidgets.QLineEdit:
        """Create a line edit for a string field."""
        line_edit = QtWidgets.QLineEdit()
        line_edit.setText(str(current_value) if current_value is not None else str(field_info.default_value))
        
        if field_info.tooltip:
            line_edit.setToolTip(field_info.tooltip)
        
        line_edit.editingFinished.connect(lambda c=comp, fn=field_name, le=line_edit: self._on_inspector_field_changed(c, fn, le.text()))
        return line_edit

    def _create_color_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a color editor widget for a color field."""
        color = np.array(current_value if current_value is not None else field_info.default_value, dtype=np.float32)
        if color.max() <= 1.0:
            color = (color * 255.0).astype(int)
        else:
            color = np.array(color).astype(int)
        color = np.clip(color, 0, 255)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        rows = []
        for label, idx in (("R", 0), ("G", 1), ("B", 2)):
            row = self._make_color_slider(label, int(color[idx]) if idx < len(color) else 128, 
                                          lambda value, c=comp, fn=field_name, w=widget: self._on_color_field_changed(c, fn, w))
            layout.addWidget(row)
            rows.append(row)
        widget._color_rows = rows
        
        if field_info.tooltip:
            widget.setToolTip(field_info.tooltip)
        
        return widget

    def _create_vector3_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a vector3 editor widget for a vector3 field."""
        value = current_value if current_value is not None else field_info.default_value
        
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = []
        
        for i, val in enumerate(value):
            spin = self._make_spinbox(
                field_info.min_value if field_info.min_value is not None else -10000.0,
                field_info.max_value if field_info.max_value is not None else 10000.0,
                field_info.step if field_info.step is not None else 0.1,
                field_info.decimals if field_info.decimals is not None else 2
            )
            spin.setValue(float(val))
            spin.valueChanged.connect(lambda v, c=comp, fn=field_name, w=widget: self._on_vector3_field_changed(c, fn, w))
            layout.addWidget(spin)
            fields.append(spin)
        
        widget._vector_fields = fields
        
        if field_info.tooltip:
            widget.setToolTip(field_info.tooltip)
        
        return widget

    def _create_enum_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QComboBox:
        """Create a combo box for an enum field."""
        combo = QtWidgets.QComboBox()
        
        if field_info.enum_options:
            for value, label in field_info.enum_options:
                combo.addItem(label, value)
            
            # Set current value
            if current_value is not None:
                index = combo.findData(current_value)
                if index >= 0:
                    combo.setCurrentIndex(index)
        
        if field_info.tooltip:
            combo.setToolTip(field_info.tooltip)
        
        combo.currentIndexChanged.connect(lambda idx, c=comp, fn=field_name, cb=combo: self._on_inspector_field_changed(c, fn, cb.currentData()))
        return combo

    def _create_list_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a dynamic list editor widget for a list field."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Container for list items
        items_container = QtWidgets.QWidget()
        items_layout = QtWidgets.QVBoxLayout(items_container)
        items_layout.setContentsMargins(0, 0, 0, 0)
        items_layout.setSpacing(2)
        
        # Get the current list value (default to empty list)
        current_list = list(current_value) if current_value is not None else []
        
        # Store references to item widgets for updating
        item_widgets = []
        
        def add_list_item(item_value=None, index=None):
            """Add a new item to the list UI."""
            item_widget = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(4)
            
            # Determine item type for the editor
            item_type = field_info.list_item_type
            
            # Create appropriate editor based on item type
            if item_type == float:
                editor = QtWidgets.QDoubleSpinBox()
                editor.setRange(-10000.0, 10000.0)
                editor.setSingleStep(0.1)
                editor.setDecimals(2)
                if item_value is not None:
                    editor.setValue(float(item_value))
                else:
                    editor.setValue(0.0)
            elif item_type == int:
                editor = QtWidgets.QSpinBox()
                editor.setRange(-10000, 10000)
                if item_value is not None:
                    editor.setValue(int(item_value))
                else:
                    editor.setValue(0)
            elif item_type == str:
                editor = QtWidgets.QLineEdit()
                if item_value is not None:
                    editor.setText(str(item_value))
                else:
                    editor.setText("")
            else:
                # Default to string for unknown types
                editor = QtWidgets.QLineEdit()
                if item_value is not None:
                    editor.setText(str(item_value))
                else:
                    editor.setText("")
            
            # Remove button
            remove_btn = QtWidgets.QPushButton("-")
            remove_btn.setFixedWidth(24)
            remove_btn.setToolTip("Remove this item")
            
            item_layout.addWidget(editor, 1)
            item_layout.addWidget(remove_btn)
            
            items_layout.addWidget(item_widget)
            item_widgets.append((item_widget, editor))
            
            # Connect remove button
            remove_btn.clicked.connect(lambda: remove_list_item(item_widget))
            
            # Connect value change to update the list
            if isinstance(editor, QtWidgets.QDoubleSpinBox):
                editor.valueChanged.connect(lambda: update_list_value())
            elif isinstance(editor, QtWidgets.QSpinBox):
                editor.valueChanged.connect(lambda: update_listValue())
            elif isinstance(editor, QtWidgets.QLineEdit):
                editor.editingFinished.connect(lambda: update_list_value())
            
            update_list_value()
            return item_widget
        
        def remove_list_item(item_widget):
            """Remove an item from the list UI."""
            for i, (widget, editor) in enumerate(item_widgets):
                if widget is item_widget:
                    item_widgets.pop(i)
                    widget.setParent(None)
                    widget.deleteLater()
                    break
            update_list_value()
        
        def update_list_value():
            """Update the component's list value from the UI."""
            new_list = []
            item_type = field_info.list_item_type
            
            for widget, editor in item_widgets:
                if isinstance(editor, QtWidgets.QDoubleSpinBox):
                    new_list.append(editor.value())
                elif isinstance(editor, QtWidgets.QSpinBox):
                    new_list.append(editor.value())
                elif isinstance(editor, QtWidgets.QLineEdit):
                    if item_type == int:
                        try:
                            new_list.append(int(editor.text()))
                        except ValueError:
                            new_list.append(0)
                    elif item_type == float:
                        try:
                            new_list.append(float(editor.text()))
                        except ValueError:
                            new_list.append(0.0)
                    else:
                        new_list.append(editor.text())
            
            comp.set_inspector_field_value(field_name, new_list)
            self._mark_scene_dirty()
        
        # Add existing items
        for item in current_list:
            add_list_item(item)
        
        # Add button
        add_btn = QtWidgets.QPushButton("+ Add Item")
        add_btn.setToolTip("Add a new item to the list")
        add_btn.clicked.connect(lambda: add_list_item())
        
        layout.addWidget(items_container)
        layout.addWidget(add_btn)
        
        # Store references for later updates
        widget._items_container = items_container
        widget._item_widgets = item_widgets
        widget._add_item_func = add_list_item
        widget._update_list_value = update_list_value
        
        if field_info.tooltip:
            widget.setToolTip(field_info.tooltip)
        
        return widget

    def _create_component_ref_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QComboBox:
        """Create a combo box for selecting a component reference."""
        combo = QtWidgets.QComboBox()
        
        # Add "None" option
        combo.addItem("(None)", None)
        
        # Get the component type filter
        component_type = field_info.component_type
        
        # Collect all components of the specified type from all game objects
        component_entries = []  # (display_name, component_instance)
        
        if self._scene:
            for obj in self._scene.objects:
                if component_type:
                    # Find components of the specified type
                    components = obj.get_components(component_type)
                    for c in components:
                        display_name = f"{obj.name} ({c.__class__.__name__})"
                        component_entries.append((display_name, c))
                else:
                    # If no specific type, show all components
                    for c in obj.components:
                        if c is not obj.transform:  # Skip transform
                            display_name = f"{obj.name} ({c.__class__.__name__})"
                            component_entries.append((display_name, c))
        
        # Sort entries by display name
        component_entries.sort(key=lambda x: x[0])
        
        # Add to combo box
        for display_name, component in component_entries:
            combo.addItem(display_name, id(component))  # Use id() as data
        
        # Set current value if there is one
        if current_value is not None:
            target_id = id(current_value)
            for i in range(combo.count()):
                if combo.itemData(i) == target_id:
                    combo.setCurrentIndex(i)
                    break
        
        if field_info.tooltip:
            combo.setToolTip(field_info.tooltip)
        
        # Handle selection change
        def on_selection_changed(idx):
            data = combo.itemData(idx)
            if data is None:
                comp.set_inspector_field_value(field_name, None)
            else:
                # Find the component by id
                for display_name, component in component_entries:
                    if id(component) == data:
                        comp.set_inspector_field_value(field_name, component)
                        break
            self._viewport.update()
            self._mark_scene_dirty()
        
        combo.currentIndexChanged.connect(on_selection_changed)
        return combo

    def _on_inspector_field_changed(self, comp, field_name: str, value: Any) -> None:
        """Handle when an inspector field value changes."""
        comp.set_inspector_field_value(field_name, value)
        self._viewport.update()
        self._mark_scene_dirty()

    def _on_color_field_changed(self, comp, field_name: str, widget: QtWidgets.QWidget) -> None:
        """Handle when a color field value changes."""
        if widget is None:
            return
        channels = []
        for row in widget._color_rows:
            row._value_label.setText(str(row._color_slider.value()))
            channels.append(row._color_slider.value() / 255.0)
        comp.set_inspector_field_value(field_name, tuple(channels))
        self._viewport.update()
        self._mark_scene_dirty()

    def _on_vector3_field_changed(self, comp, field_name: str, widget: QtWidgets.QWidget) -> None:
        """Handle when a vector3 field value changes."""
        if widget is None:
            return
        values = tuple(field.value() for field in widget._vector_fields)
        comp.set_inspector_field_value(field_name, values)
        
        # Special handling for collider center changes
        from src.physics.collider import Collider
        if isinstance(comp, Collider):
            comp._transform_dirty = True
        
        self._viewport.update()
        self._mark_scene_dirty()

    def _refresh_component_fields(self, obj: GameObject) -> None:
        from src.engine3d.light import Light3D
        from src.physics.collider import Collider
        from src.engine3d.object3d import Object3D
        from src.physics.rigidbody import Rigidbody

        component_boxes = [
            box for box in self._component_fields
            if isinstance(box, QtWidgets.QGroupBox)
        ]
        comp_index = 0

        non_transform_components = [comp for comp in obj.components if comp is not obj.transform]
        if len(non_transform_components) != len(component_boxes):
            self._components_dirty = True
            self._build_component_fields(obj)
            return

        for comp_index, comp in enumerate(non_transform_components):
            box = component_boxes[comp_index] if comp_index < len(component_boxes) else None
            if box is None:
                self._components_dirty = True
                self._build_component_fields(obj)
                return

            if getattr(box, "_component_ref", None) is not comp:
                self._components_dirty = True
                self._build_component_fields(obj)
                return

            self._update_component_box_title(box, comp.__class__.__name__)

            # Check if the component uses the new inspector field system
            if hasattr(box, "_inspector_field_widgets"):
                self._refresh_inspector_field_widgets(box, comp)
            elif isinstance(comp, Light3D):
                self._refresh_light_fields(box, comp)
            elif isinstance(comp, Collider):
                self._refresh_collider_fields(box, comp)
            elif isinstance(comp, Object3D):
                self._refresh_object3d_fields(box, comp)
            elif isinstance(comp, Rigidbody):
                self._refresh_rigidbody_fields(box, comp)

        if comp_index + 1 != len(component_boxes):
            self._components_dirty = True
            self._build_component_fields(obj)

    def _refresh_inspector_field_widgets(self, box: QtWidgets.QGroupBox, comp) -> None:
        """Refresh the values of inspector field widgets for a component."""
        field_widgets = getattr(box, "_inspector_field_widgets", {})
        
        for field_name, widget in field_widgets.items():
            current_value = comp.get_inspector_field_value(field_name)
            
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                if not widget.hasFocus():
                    widget.setValue(float(current_value) if current_value is not None else 0.0)
            elif isinstance(widget, QtWidgets.QSpinBox):
                if not widget.hasFocus():
                    widget.setValue(int(current_value) if current_value is not None else 0)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(current_value) if current_value is not None else False)
            elif isinstance(widget, QtWidgets.QLineEdit):
                if not widget.hasFocus():
                    widget.setText(str(current_value) if current_value is not None else "")
            elif hasattr(widget, "_color_rows"):
                # Color widget
                self._refresh_color_editor(widget, current_value)
            elif hasattr(widget, "_vector_fields"):
                # Vector3 widget
                self._refresh_vector_row(widget, current_value)
            # Note: List and component_ref widgets are not refreshed dynamically
            # as they are complex UI structures that would be rebuilt on selection change

    def _create_object3d_fields(self, comp: 'Object3D') -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)

        color_widget = self._create_color_editor(comp)
        form_layout.addRow("Color", color_widget)
        box._color_widget = color_widget
        
        main_layout.addLayout(form_layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box

    def _refresh_object3d_fields(self, box: QtWidgets.QGroupBox, comp: 'Object3D') -> None:
        if hasattr(box, "_color_widget"):
            self._refresh_color_editor(box._color_widget, comp.color)

    def _create_rigidbody_fields(self, comp: 'Rigidbody') -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        use_gravity = QtWidgets.QCheckBox()
        use_gravity.setChecked(comp.use_gravity)
        use_gravity.toggled.connect(lambda val, c=comp: setattr(c, "use_gravity", val))
        layout.addRow("Use Gravity", use_gravity)
        box._use_gravity_field = use_gravity

        is_kinematic = QtWidgets.QCheckBox()
        is_kinematic.setChecked(comp.is_kinematic)
        is_kinematic.toggled.connect(lambda val, c=comp: setattr(c, "is_kinematic", val))
        layout.addRow("Is Kinematic", is_kinematic)
        box._is_kinematic_field = is_kinematic

        is_static = QtWidgets.QCheckBox()
        is_static.setChecked(comp.is_static)
        is_static.toggled.connect(lambda val, c=comp: setattr(c, "is_static", val))
        layout.addRow("Is Static", is_static)
        box._is_static_field = is_static

        mass = self._make_spinbox(0.001, 10000.0, step=0.1, decimals=2)
        mass.setValue(float(comp.mass))
        mass.valueChanged.connect(lambda val, c=comp: setattr(c, "mass", float(val)))
        layout.addRow("Mass", mass)
        box._mass_field = mass

        drag = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        drag.setValue(float(comp.drag))
        drag.valueChanged.connect(lambda val, c=comp: setattr(c, "drag", float(val)))
        layout.addRow("Drag", drag)
        box._drag_field = drag

        main_layout.addLayout(layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)

        return box

    def _refresh_rigidbody_fields(self, box: QtWidgets.QGroupBox, comp: 'Rigidbody') -> None:
        if hasattr(box, "_use_gravity_field"):
            box._use_gravity_field.setChecked(comp.use_gravity)
        if hasattr(box, "_is_kinematic_field"):
            box._is_kinematic_field.setChecked(comp.is_kinematic)
        if hasattr(box, "_is_static_field"):
            box._is_static_field.setChecked(comp.is_static)
        if hasattr(box, "_mass_field"):
            self._apply_spinbox(box._mass_field, float(comp.mass))
        if hasattr(box, "_drag_field"):
            self._apply_spinbox(box._drag_field, float(comp.drag))

    def _create_component_summary(self, comp) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        label = QtWidgets.QLabel("No editable fields")
        label.setStyleSheet("color: #888;")
        layout.addWidget(label)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        layout.addWidget(remove_btn)
        
        return box

    def _create_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(light.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        intensity = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        intensity.setValue(float(light.intensity))
        intensity.valueChanged.connect(lambda value, l=light: self._on_light_intensity_changed(l, value))
        layout.addRow("Intensity", intensity)
        box._intensity_field = intensity

        color_widget = self._create_color_editor(light)
        layout.addRow("Color", color_widget)
        box._color_widget = color_widget
        
        main_layout.addLayout(layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=light: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box

    def _create_directional_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = self._create_light_fields(light)
        layout = box.layout()

        ambient = self._make_spinbox(0.0, 1.0, step=0.05, decimals=2)
        ambient.setValue(float(light.ambient))
        ambient.valueChanged.connect(lambda value, l=light: self._on_directional_light_ambient_changed(l, value))
        layout.addRow("Ambient", ambient)
        box._ambient_field = ambient
        return box

    def _create_point_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = self._create_light_fields(light)
        layout = box.layout()

        range_field = self._make_spinbox(0.1, 1000.0, step=0.5, decimals=2)
        range_field.setValue(float(light.range))
        range_field.valueChanged.connect(lambda value, l=light: self._on_point_light_range_changed(l, value))
        layout.addRow("Range", range_field)
        box._range_field = range_field
        return box

    def _create_color_editor(self, light) -> QtWidgets.QWidget:
        color = np.array(light.color, dtype=np.float32)
        if color.max() <= 1.0:
            color = (color * 255.0).astype(int)
        else:
            color = np.array(color).astype(int)
        color = np.clip(color, 0, 255)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        rows = []
        for label, idx in (("R", 0), ("G", 1), ("B", 2)):
            row = self._make_color_slider(label, int(color[idx]), lambda value, l=light, w=widget: self._on_light_color_changed(l, w))
            layout.addWidget(row)
            rows.append(row)
        widget._color_rows = rows
        return widget

    def _create_collider_fields(self, collider) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(collider.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        center_row = self._make_vector_row(collider.center, lambda value, c=collider: self._on_collider_center_changed(c, center_row))
        layout.addRow("Center", center_row)
        box._center_row = center_row
        
        main_layout.addLayout(layout)
        
        # Store the form layout for subclasses to add more fields
        box._form_layout = layout
        box._main_layout = main_layout
        
        return box

    def _create_box_collider_fields(self, collider: 'BoxCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box._form_layout

        size_row = self._make_vector_row(collider.size, lambda value, c=collider: self._on_box_collider_size_changed(c, size_row))
        layout.addRow("Size", size_row)
        box._size_row = size_row
        
        # Add remove button at the end
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=collider: self._remove_component(c))
        box._main_layout.addWidget(remove_btn)
        
        return box

    def _create_sphere_collider_fields(self, collider: 'SphereCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box._form_layout

        radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        radius.setValue(float(collider.radius))
        radius.valueChanged.connect(lambda value, c=collider: self._on_sphere_collider_radius_changed(c, value))
        layout.addRow("Radius", radius)
        box._radius_field = radius
        
        # Add remove button at the end
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=collider: self._remove_component(c))
        box._main_layout.addWidget(remove_btn)
        
        return box

    def _create_capsule_collider_fields(self, collider: 'CapsuleCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box._form_layout

        radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        radius.setValue(float(collider.radius))
        radius.valueChanged.connect(lambda value, c=collider: self._on_capsule_collider_radius_changed(c, value))
        layout.addRow("Radius", radius)

        height = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        height.setValue(float(collider.height))
        height.valueChanged.connect(lambda value, c=collider: self._on_capsule_collider_height_changed(c, value))
        layout.addRow("Height", height)

        box._radius_field = radius
        box._height_field = height
        
        # Add remove button at the end
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=collider: self._remove_component(c))
        box._main_layout.addWidget(remove_btn)
        
        return box

    def _refresh_light_fields(self, box: QtWidgets.QGroupBox, light) -> None:
        if hasattr(box, "_intensity_field"):
            self._apply_spinbox(box._intensity_field, float(light.intensity))
        if hasattr(box, "_ambient_field") and hasattr(light, "ambient"):
            self._apply_spinbox(box._ambient_field, float(light.ambient))
        if hasattr(box, "_range_field") and hasattr(light, "range"):
            self._apply_spinbox(box._range_field, float(light.range))
        if hasattr(box, "_color_widget"):
            self._refresh_color_editor(box._color_widget, light.color)

    def _refresh_color_editor(self, widget: QtWidgets.QWidget, color_value) -> None:
        color = np.array(color_value, dtype=np.float32)
        if color.max() <= 1.0:
            color = (color * 255.0).astype(int)
        else:
            color = np.array(color).astype(int)
        color = np.clip(color, 0, 255)
        for idx, row in enumerate(widget._color_rows):
            self._apply_slider(row._color_slider, int(color[idx]))
            row._value_label.setText(str(int(color[idx])))

    def _refresh_collider_fields(self, box: QtWidgets.QGroupBox, collider) -> None:
        if hasattr(box, "_center_row"):
            self._refresh_vector_row(box._center_row, collider.center)
        if hasattr(box, "_size_row") and hasattr(collider, "size"):
            self._refresh_vector_row(box._size_row, collider.size)
        if hasattr(box, "_radius_field") and hasattr(collider, "radius"):
            self._apply_spinbox(box._radius_field, float(collider.radius))
        if hasattr(box, "_height_field") and hasattr(collider, "height"):
            self._apply_spinbox(box._height_field, float(collider.height))

    def _refresh_vector_row(self, row_widget: QtWidgets.QWidget, values: Iterable[float]) -> None:
        fields = getattr(row_widget, "_vector_fields", [])
        for idx, value in enumerate(values):
            if idx < len(fields):
                self._apply_spinbox(fields[idx], float(value))

    def _on_light_intensity_changed(self, light, value: float) -> None:
        light.intensity = float(value)
        self._viewport.update()

    def _on_directional_light_ambient_changed(self, light, value: float) -> None:
        light.ambient = float(value)
        self._viewport.update()

    def _on_point_light_range_changed(self, light, value: float) -> None:
        light.range = float(value)
        self._viewport.update()

    def _on_light_color_changed(self, light, widget: QtWidgets.QWidget) -> None:
        if widget is None:
            return
        self._apply_light_color_from_widget(light, widget)

    def _apply_light_color_from_widget(self, light, widget: QtWidgets.QWidget) -> None:
        channels = []
        for row in widget._color_rows:
            row._value_label.setText(str(row._color_slider.value()))
            channels.append(row._color_slider.value() / 255.0)
        light.color = tuple(channels)
        self._viewport.update()

    def _on_collider_center_changed(self, collider, row_widget: QtWidgets.QWidget) -> None:
        values = [field.value() for field in row_widget._vector_fields]
        collider.center = values
        collider._transform_dirty = True
        self._viewport.update()

    def _on_box_collider_size_changed(self, collider: 'BoxCollider', row_widget: QtWidgets.QWidget) -> None:
        values = [field.value() for field in row_widget._vector_fields]
        collider.size = values
        collider._transform_dirty = True
        self._viewport.update()

    def _on_sphere_collider_radius_changed(self, collider: 'SphereCollider', value: float) -> None:
        collider.radius = float(value)
        collider._transform_dirty = True
        self._viewport.update()

    def _on_capsule_collider_radius_changed(self, collider: 'CapsuleCollider', value: float) -> None:
        collider.radius = float(value)
        collider._transform_dirty = True
        self._viewport.update()

    def _on_capsule_collider_height_changed(self, collider: 'CapsuleCollider', value: float) -> None:
        collider.height = float(value)
        collider._transform_dirty = True
        self._viewport.update()
