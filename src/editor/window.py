from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

from src.engine3d.scene import Scene3D
from src.engine3d.window import Window3D
from src.engine3d.gameobject import GameObject
from src.engine3d.object3d import create_cube, create_sphere, create_plane, Object3D

from .selection import EditorSelection
from .viewport import ViewportWidget
from .scene import EditorScene


class HierarchyTreeWidget(QtWidgets.QTreeWidget):
    """Custom tree widget that supports drag-drop parenting of GameObjects."""
    object_parented = QtCore.Signal(object, object)  # (child_obj, parent_obj or None)
    
    def __init__(self, editor_window, parent=None):
        super().__init__(parent)
        self.editor_window = editor_window
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
    
    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle drop event to parent objects."""
        # Get the item being dragged
        dragged_item = self.currentItem()
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
        
        # Emit signal for the parenting operation
        # If drop_obj is None, it means dropping at root level
        self.object_parented.emit(dragged_obj, drop_obj)
        
        # Accept the event
        event.acceptProposedAction()
    
    def _is_descendant(self, potential_ancestor: GameObject, potential_descendant: GameObject) -> bool:
        """Check if potential_descendant is a descendant of potential_ancestor."""
        current = potential_descendant.transform.parent
        while current:
            if current.game_object is potential_ancestor:
                return True
            current = current.parent
        return False


class EditorWindow(QtWidgets.QMainWindow):
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

        self._build_layout()
        self._setup_files_panel()
        self._setup_hierarchy_panel()
        self._setup_inspector_panel()
        self._setup_toolbar()
        self._setup_timer()
        self._setup_camera_controls()

        QtCore.QTimer.singleShot(0, self._init_engine)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
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

        button_row = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add", panel)
        remove_button = QtWidgets.QPushButton("Remove", panel)
        add_button.clicked.connect(self._show_add_menu)
        remove_button.clicked.connect(self._remove_selected)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
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
        self._select_object(child_obj)
        self._viewport.update()
        self._viewport.doneCurrent()

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
            "Script": self._add_script_component,
        }

        for name, callback in actions.items():
            action = menu.addAction(name)
            action.triggered.connect(callback)

        menu.exec(QtGui.QCursor.pos())

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
        script_template = f'''from src.engine3d import Script, Time


class {class_name}(Script):
    """
    Custom script component.
    """
    
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

            # Load the module
            spec = importlib.util.spec_from_file_location(
                file_path.stem, str(file_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load script from {file_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path.stem] = module
            spec.loader.exec_module(module)

            # Get the class from the module
            if not hasattr(module, class_name):
                raise AttributeError(f"Script file does not contain class '{class_name}'")

            script_class = getattr(module, class_name)
            script_instance = script_class()

            # Add to selected game object
            self._add_component_to_selected(script_instance)

        except Exception as e:
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
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to start play mode:\n{e}")

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

    def _render_frame(self) -> None:
        """Called by ViewportWidget.paintGL() to render the frame."""
        if not self._window:
            return
            
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
            
            # Right vector: perpendicular to forward (in XZ plane) and world up
            # At azimuth 0, forward is -Z, so right is +X
            # At azimuth 90, forward is +X, so right is +Z
            right = np.array([
                -np.cos(azimuth_rad),  # X component
                0.0,                    # Y component (horizontal right)
                np.sin(azimuth_rad)     # Z component
            ], dtype=np.float32)
            
            # Up vector: world up (0, 1, 0) for horizontal panning
            # For true camera-relative up, we'd need to account for elevation
            # But for panning, we want to move perpendicular to the view direction
            # Project world_up onto the plane perpendicular to forward
            forward = np.array([
                -np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
                np.cos(elevation_rad) * np.cos(azimuth_rad)
            ], dtype=np.float32)
            forward = forward / np.linalg.norm(forward)
            
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # True up vector relative to camera orientation
            up = np.cross(right, forward)
            up_norm = np.linalg.norm(up)
            if up_norm > 0.001:
                up = up / up_norm
            else:
                up = world_up
            
            # Pan target
            # dx < 0 moves target left (camera appears to move right), so add right * (-dx)
            # dy < 0 moves target up (camera appears to move down), so add up * dy
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
        """Update camera position based on spherical coordinates.
        
        Coordinate system:
        - Y is up
        - Azimuth 0 = looking along -Z (into the screen)
        - Azimuth 90 = looking along +X (right)
        - Azimuth 180 = looking along +Z (towards camera)
        - Elevation 0 = horizontal
        - Elevation 90 = looking straight up (+Y)
        - Elevation -90 = looking straight down (-Y)
        """
        if not self._window:
            return

        azimuth_rad = np.radians(self._camera_control['azimuth'])
        elevation_rad = np.radians(self._camera_control['elevation'])
        distance = self._camera_control['distance']
        target = self._camera_control['target']

        # Calculate camera position on sphere around target
        # Using right-handed coordinate system:
        # X = right, Y = up, Z = back (away from camera at azimuth=0)
        # Azimuth 0: looking along -Z (into screen), camera is at +Z
        # Azimuth 90: looking along +X, camera is at -X
        cam_offset = np.array([
            -distance * np.cos(elevation_rad) * np.sin(azimuth_rad),  # X: -sin for right-handed
            distance * np.sin(elevation_rad),                          # Y: up
            distance * np.cos(elevation_rad) * np.cos(azimuth_rad)     # Z: +cos
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
        
        # Expand all items to show hierarchy
        self._hierarchy_tree.expandAll()

    def _show_add_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        empty_action = menu.addAction("Empty GameObject")
        cube_action = menu.addAction("Cube")
        sphere_action = menu.addAction("Sphere")
        plane_action = menu.addAction("Plane")
        camera_action = menu.addAction("Camera")
        
        action = menu.exec(QtGui.QCursor.pos())
        if action == empty_action:
            self._add_object(GameObject(), "GameObject")
        elif action == cube_action:
            self._add_object(create_cube(1.0), "Cube")
        elif action == sphere_action:
            self._add_object(create_sphere(0.75), "Sphere")
        elif action == plane_action:
            self._add_object(create_plane(5.0, 5.0), "Plane")
        elif action == camera_action:
            from src.engine3d.camera import Camera3D
            go = GameObject("Camera")
            go.add_component(Camera3D())
            self._add_object(go, "Camera")

    def _add_object(self, obj: GameObject, name: str) -> None:
        self._viewport.makeCurrent()
        obj.name = name
        self._scene.add_object(obj)
        self._refresh_hierarchy()
        self._select_object(obj)
        self._viewport.update()
        self._viewport.doneCurrent()

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
            for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
                for f in fields:
                    f.setValue(0.0)
            self._clear_component_fields()
            self._components_dirty = True
            return

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
            if isinstance(comp, Light3D):
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

            if isinstance(comp, Light3D):
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

    def _create_object3d_fields(self, comp: 'Object3D') -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        layout = QtWidgets.QFormLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        color_widget = self._create_color_editor(comp)
        layout.addRow("Color", color_widget)
        box._color_widget = color_widget
        return box

    def _refresh_object3d_fields(self, box: QtWidgets.QGroupBox, comp: 'Object3D') -> None:
        if hasattr(box, "_color_widget"):
            self._refresh_color_editor(box._color_widget, comp.color)

    def _create_rigidbody_fields(self, comp: 'Rigidbody') -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        layout = QtWidgets.QFormLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

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
        return box

    def _create_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(light.__class__.__name__)
        layout = QtWidgets.QFormLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        intensity = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        intensity.setValue(float(light.intensity))
        intensity.valueChanged.connect(lambda value, l=light: self._on_light_intensity_changed(l, value))
        layout.addRow("Intensity", intensity)
        box._intensity_field = intensity

        color_widget = self._create_color_editor(light)
        layout.addRow("Color", color_widget)
        box._color_widget = color_widget
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
        layout = QtWidgets.QFormLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        center_row = self._make_vector_row(collider.center, lambda value, c=collider: self._on_collider_center_changed(c, center_row))
        layout.addRow("Center", center_row)
        box._center_row = center_row
        return box

    def _create_box_collider_fields(self, collider: 'BoxCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box.layout()

        size_row = self._make_vector_row(collider.size, lambda value, c=collider: self._on_box_collider_size_changed(c, size_row))
        layout.addRow("Size", size_row)
        box._size_row = size_row
        return box

    def _create_sphere_collider_fields(self, collider: 'SphereCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box.layout()

        radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        radius.setValue(float(collider.radius))
        radius.valueChanged.connect(lambda value, c=collider: self._on_sphere_collider_radius_changed(c, value))
        layout.addRow("Radius", radius)
        box._radius_field = radius
        return box

    def _create_capsule_collider_fields(self, collider: 'CapsuleCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box.layout()

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
