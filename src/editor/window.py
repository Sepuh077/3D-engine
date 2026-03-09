from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

from src.engine3d.scene import Scene3D
from src.engine3d.window import Window3D
from src.engine3d.gameobject import GameObject
from src.engine3d.object3d import create_cube, create_sphere, create_plane

from .selection import EditorSelection
from .viewport import ViewportWidget
from .scene import EditorScene

class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self, project_root: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.setWindowTitle("Engine3D Editor")
        self.resize(1280, 768)

        self._selection = EditorSelection()
        self._scene = EditorScene()
        self._window: Optional[Window3D] = None
        self._scene_auto_objects = {"Main Camera", "Directional Light"}
        self._object_items: Dict[GameObject, QtWidgets.QTreeWidgetItem] = {}
        self._component_fields: list[QtWidgets.QWidget] = []
        self._components_dirty = True

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
        toolbar = QtWidgets.QToolBar("Transform", self)
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

    def _add_toolbar_button(self, toolbar: QtWidgets.QToolBar, label: str, callback) -> None:
        action = QtGui.QAction(label, self)
        action.triggered.connect(callback)
        toolbar.addAction(action)

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

        self._hierarchy_tree = QtWidgets.QTreeWidget(panel)
        self._hierarchy_tree.setHeaderLabel("GameObjects")
        self._hierarchy_tree.itemSelectionChanged.connect(self._on_hierarchy_selection)
        self._hierarchy_tree.itemDoubleClicked.connect(self._on_hierarchy_double_click)
        layout.addWidget(self._hierarchy_tree)

        self._hierarchy_dock.setWidget(panel)

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
        }

        for name, callback in actions.items():
            action = menu.addAction(name)
            action.triggered.connect(callback)

        menu.exec(QtGui.QCursor.pos())

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
        layout.addWidget(self._file_view)

        self._files_dock.setWidget(panel)

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
        self._window.editor_show_camera = False  # Hide the camera frustum visualization
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
            
        if not self._window.tick(simulate=False):
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

    def _on_mouse_pressed(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse button press for camera control."""
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
        
        # Update scene camera instead of window camera (since scene is active)
        self._scene.camera.position = tuple(camera_pos)
        self._scene.camera.look_at(tuple(target))
        self._viewport.update()

    def _refresh_hierarchy(self) -> None:
        self._hierarchy_tree.clear()
        self._object_items.clear()
        for obj in self._scene.objects:
            if obj.name in self._scene_auto_objects:
                continue
            item = QtWidgets.QTreeWidgetItem([obj.name])
            self._hierarchy_tree.addTopLevelItem(item)
            self._object_items[obj] = item

    def _show_add_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        cube_action = menu.addAction("Cube")
        sphere_action = menu.addAction("Sphere")
        plane_action = menu.addAction("Plane")
        action = menu.exec(QtGui.QCursor.pos())
        if action == cube_action:
            self._add_object(create_cube(1.0), "Cube")
        elif action == sphere_action:
            self._add_object(create_sphere(0.75), "Sphere")
        elif action == plane_action:
            self._add_object(create_plane(5.0, 5.0), "Plane")

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
            else:
                box = self._create_component_summary(comp)
            box._component_ref = comp
            self._ensure_component_box(box)

        self._components_dirty = False

    def _refresh_component_fields(self, obj: GameObject) -> None:
        from src.engine3d.light import Light3D
        from src.physics.collider import Collider

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

        if comp_index + 1 != len(component_boxes):
            self._components_dirty = True
            self._build_component_fields(obj)

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
