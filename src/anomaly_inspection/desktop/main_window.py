from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from anomaly_inspection.desktop.pages.project_setup import ProjectSetupPage
from anomaly_inspection.desktop.ui.icons import apply_window_icon
from anomaly_inspection.desktop.pages.inspect_camera import InspectCameraPage
from anomaly_inspection.desktop.pages.inspect_image import InspectImagePage
from anomaly_inspection.desktop.pages.logs import LogsPage
from anomaly_inspection.desktop.pages.reference_capture import ReferenceCapturePage
from anomaly_inspection.desktop.runtime import PreparedRuntimeManager
from anomaly_inspection.desktop.state import AppState
from anomaly_inspection.desktop.ui.theme import (
    apply_app_theme,
    available_theme_names,
    resolve_theme,
    theme_dimensions,
    theme_display_name,
    theme_spacing,
    zero_margins,
)
from anomaly_inspection.desktop.ui.components import set_button_icon
from anomaly_inspection.desktop.ui.locale import apply_app_locale
from anomaly_inspection.desktop.pages.zone_editor import ZoneEditorPage


@dataclass(frozen=True)
class PageSpec:
    title: str
    description: str
    scroll_policy: str = "page"


PAGE_SPECS: tuple[PageSpec, ...] = (
    PageSpec("Project Setup", "Prepare the inspection job and required files."),
    PageSpec("Capture Reference", "Capture the empty station reference."),
    PageSpec("Draw Zones", "Define presence-check regions."),
    PageSpec("Inspect Image", "Inspect a saved image.", scroll_policy="workbench"),
    PageSpec("Inspect Camera", "Capture and inspect a camera frame.", scroll_policy="workbench"),
    PageSpec("Logs", "Review inspection history and artifacts."),
)

NAV_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Setup", (0, 1, 2)),
    ("Inspection", (3, 4)),
    ("Review", (5,)),
)


class MainWindow(QMainWindow):
    COMPACT_SHELL_WIDTH = 1120
    COMPACT_NAV_LABELS = {
        0: "Setup",
        1: "Reference",
        2: "Zones",
        3: "Image",
        4: "Camera",
        5: "Logs",
    }

    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        self.runtime_manager = PreparedRuntimeManager()
        self.state.desktop_theme_name = resolve_theme(self.state.desktop_theme_name).name
        app = QApplication.instance()
        if app is not None:
            apply_app_theme(app, self.state.desktop_theme_name)

        self.setWindowTitle("Anomaly Inspection Pipeline")
        apply_window_icon(self)
        dimensions = theme_dimensions()
        self._stack = QStackedWidget()
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._project_setup_page: ProjectSetupPage | None = None
        self._reference_capture_page: ReferenceCapturePage | None = None
        self._zone_editor_page: ZoneEditorPage | None = None
        self._inspect_image_page: InspectImagePage | None = None
        self._inspect_camera_page: InspectCameraPage | None = None
        self._logs_page: LogsPage | None = None
        self.theme_combo: QComboBox | None = None
        self._nav_buttons: dict[int, QPushButton] = {}
        self._nav_group_labels: list[QLabel] = []
        self._nav_separators: list[QFrame] = []
        self._compact_shell = False

        self._build_ui()
        apply_app_locale(self)
        self.resize(dimensions.window_initial_width, dimensions.window_initial_height)
        self.setMinimumSize(dimensions.window_min_width, dimensions.window_min_height)
        self._update_shell_compact_mode(self.width())

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(*zero_margins())
        root_layout.setSpacing(0)

        top_nav = self._build_top_navigation()
        content = self._build_content()

        root_layout.addWidget(top_nav)
        root_layout.addWidget(content, 1)

        self.setCentralWidget(root)

    def _build_top_navigation(self) -> QWidget:
        header = QFrame()
        header.setObjectName("topNavShell")
        header.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(header)
        spacing = theme_spacing()
        layout.setContentsMargins(
            spacing.shell_margin_horizontal,
            spacing.shell_margin_vertical,
            spacing.shell_margin_horizontal,
            spacing.shell_margin_vertical,
        )
        layout.setSpacing(spacing.shell_gap)

        brand_area = QWidget()
        brand_area.setObjectName("topBrandArea")
        self.brand_area = brand_area
        brand_area.setMinimumWidth(190)
        brand_layout = QVBoxLayout(brand_area)
        brand_layout.setContentsMargins(*zero_margins())
        brand_layout.setSpacing(0)

        app_title = QLabel("Inspection")
        app_title.setObjectName("topNavTitle")
        brand_layout.addWidget(app_title)

        app_subtitle = QLabel("Production workstation")
        app_subtitle.setObjectName("topNavSubtitle")
        self.brand_subtitle = app_subtitle
        brand_layout.addWidget(app_subtitle)
        layout.addWidget(brand_area, 0, Qt.AlignmentFlag.AlignVCenter)

        context_area = QWidget()
        context_area.setObjectName("topPageContextArea")
        context_area.setMinimumWidth(180)
        context_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        context_layout = QVBoxLayout(context_area)
        context_layout.setContentsMargins(*zero_margins())
        context_layout.setSpacing(0)

        self.page_context_title = QLabel()
        self.page_context_title.setObjectName("topPageContextTitle")
        context_layout.addWidget(self.page_context_title)

        self.page_context_subtitle = QLabel()
        self.page_context_subtitle.setObjectName("topPageContextSubtitle")
        self.page_context_subtitle.setWordWrap(False)
        context_layout.addWidget(self.page_context_subtitle)
        layout.addWidget(context_area, 0, Qt.AlignmentFlag.AlignVCenter)

        nav_cluster = QWidget()
        nav_cluster.setObjectName("topNavCluster")
        nav_row = QHBoxLayout(nav_cluster)
        nav_row.setContentsMargins(*zero_margins())
        nav_row.setSpacing(spacing.nav_cluster_gap)

        for group_position, (group_title, page_indexes) in enumerate(NAV_GROUPS):
            group_widget = QWidget()
            group_widget.setObjectName("topNavGroup")
            group_layout = QHBoxLayout(group_widget)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(spacing.nav_cluster_gap)

            label = QLabel(group_title)
            label.setObjectName("navGroupLabel")
            self._nav_group_labels.append(label)
            group_layout.addWidget(label)
            for index in page_indexes:
                page = PAGE_SPECS[index]
                button = QPushButton(page.title)
                button.setObjectName("navButton")
                button.setCheckable(True)
                button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                button.clicked.connect(lambda checked=False, i=index: self._set_page(i))
                self._button_group.addButton(button, index)
                self._nav_buttons[index] = button
                group_layout.addWidget(button)
            nav_row.addWidget(group_widget, 0, Qt.AlignmentFlag.AlignVCenter)
            if group_position < len(NAV_GROUPS) - 1:
                separator = QFrame()
                separator.setObjectName("topNavSeparator")
                separator.setFrameShape(QFrame.Shape.VLine)
                self._nav_separators.append(separator)
                nav_row.addWidget(separator)
        nav_row.addStretch(1)
        layout.addWidget(nav_cluster, 1, Qt.AlignmentFlag.AlignVCenter)

        theme_area = QWidget()
        theme_area.setObjectName("topThemeArea")
        theme_layout = QHBoxLayout(theme_area)
        theme_layout.setContentsMargins(*zero_margins())
        theme_layout.setSpacing(spacing.nav_cluster_gap)

        theme_label = QLabel("Theme")
        theme_label.setObjectName("navGroupLabel")
        self.theme_label = theme_label
        theme_layout.addWidget(theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("themeSelector")
        self.theme_combo.setMinimumWidth(112)
        self.theme_combo.setMaximumWidth(150)
        for theme_name in available_theme_names():
            self.theme_combo.addItem(theme_display_name(theme_name), theme_name)
        theme_index = self.theme_combo.findData(self.state.desktop_theme_name)
        self.theme_combo.setCurrentIndex(theme_index if theme_index >= 0 else 0)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        theme_layout.addWidget(self.theme_combo)
        layout.addWidget(theme_area, 0, Qt.AlignmentFlag.AlignVCenter)

        first_button = self._button_group.button(0)
        if first_button is not None:
            first_button.setChecked(True)
        self._update_page_context(0)

        return header

    def _build_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("contentHost")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(*zero_margins())

        self._project_setup_page = ProjectSetupPage(self.state, self._set_status_message, self.runtime_manager)
        self._reference_capture_page = ReferenceCapturePage(
            self.state,
            self._set_status_message,
            self._on_reference_saved,
        )
        self._zone_editor_page = ZoneEditorPage(
            self.state,
            self._set_status_message,
            self._on_zones_saved,
        )
        self._inspect_image_page = InspectImagePage(self.state, self._set_status_message, self.runtime_manager)
        self._inspect_camera_page = InspectCameraPage(self.state, self._set_status_message, self.runtime_manager)
        self._logs_page = LogsPage(self.state, self._set_status_message)
        for index, page in enumerate(
            (
                self._project_setup_page,
                self._reference_capture_page,
                self._zone_editor_page,
                self._inspect_image_page,
                self._inspect_camera_page,
                self._logs_page,
            )
        ):
            self._stack.addWidget(_page_host(page, PAGE_SPECS[index].scroll_policy))

        content_layout.addWidget(self._stack)
        return content

    def _set_page(self, index: int) -> None:
        button = self._button_group.button(index)
        if button is not None:
            button.setChecked(True)
        if index == 0 and self._project_setup_page is not None:
            self._project_setup_page.refresh_from_state()
        if index == 1 and self._reference_capture_page is not None:
            self._reference_capture_page.refresh_from_state()
        if index == 2 and self._zone_editor_page is not None:
            self._zone_editor_page.refresh_from_state()
        if index == 3 and self._inspect_image_page is not None:
            self._inspect_image_page.refresh_from_state()
        if index == 4 and self._inspect_camera_page is not None:
            self._inspect_camera_page.refresh_from_state()
        if index == 5 and self._logs_page is not None:
            self._logs_page.refresh_from_state()
        self._stack.setCurrentIndex(index)
        self._update_page_context(index)

    def _update_page_context(self, index: int) -> None:
        page = PAGE_SPECS[index]
        self.page_context_title.setText(page.title)
        self.page_context_subtitle.setText(page.description)

    def _set_status_message(self, message: str) -> None:
        self.state.status_message = message

    def _on_theme_selected(self, index: int) -> None:
        if self.theme_combo is None:
            return
        theme_name = self.theme_combo.itemData(index)
        if not theme_name or theme_name == self.state.desktop_theme_name:
            return
        self.state.desktop_theme_name = str(theme_name)
        app = QApplication.instance()
        if app is not None:
            apply_app_theme(app, self.state.desktop_theme_name)
        self._refresh_theme_dependent_widgets()
        self._set_status_message(f"Theme selected: {theme_display_name(self.state.desktop_theme_name)}")

    def _refresh_theme_dependent_widgets(self) -> None:
        if self._reference_capture_page is not None:
            self._reference_capture_page.refresh_theme()
        if self._zone_editor_page is not None:
            self._zone_editor_page.refresh_theme()
        if self._inspect_camera_page is not None:
            self._inspect_camera_page.refresh_theme()
        if self._logs_page is not None:
            self._logs_page.refresh_theme()
        self._refresh_semantic_icons()

    def _refresh_semantic_icons(self) -> None:
        for widget in self.findChildren(QWidget):
            refresh = getattr(widget, "refresh_icon", None)
            if callable(refresh):
                refresh()
        for button in self.findChildren(QPushButton):
            icon_name = button.property("appIconName")
            if icon_name:
                size = button.property("appIconSize") or 14
                set_button_icon(button, str(icon_name), size=int(size))

    def _on_reference_saved(self, path) -> None:
        if self._project_setup_page is not None:
            self._project_setup_page.set_reference_image_path(path)
        if self._zone_editor_page is not None:
            self._zone_editor_page.refresh_from_state()
        self._set_status_message(f"Reference saved: {path}")

    def _on_zones_saved(self, path) -> None:
        if self._project_setup_page is not None:
            self._project_setup_page.set_zones_path(path)
        self._set_status_message(f"Zones saved: {path}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._reference_capture_page is not None:
            self._reference_capture_page.shutdown()
        if self._project_setup_page is not None:
            self._project_setup_page.shutdown()
        if self._inspect_image_page is not None:
            self._inspect_image_page.shutdown()
        if self._inspect_camera_page is not None:
            self._inspect_camera_page.shutdown()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_shell_compact_mode(event.size().width())

    def _update_shell_compact_mode(self, width: int) -> None:
        compact = width < self.COMPACT_SHELL_WIDTH
        if compact == self._compact_shell:
            return
        self._compact_shell = compact
        central = self.centralWidget()
        if central is not None:
            top_nav = central.findChild(QFrame, "topNavShell")
            if top_nav is not None:
                top_nav.setProperty("compact", compact)
                top_nav.style().unpolish(top_nav)
                top_nav.style().polish(top_nav)

        self.page_context_subtitle.setVisible(not compact)
        self.brand_area.setVisible(not compact)
        self.theme_label.setVisible(not compact)
        for label in self._nav_group_labels:
            label.setVisible(not compact)
        for separator in self._nav_separators:
            separator.setVisible(not compact)
        for index, button in self._nav_buttons.items():
            button.setText(self.COMPACT_NAV_LABELS[index] if compact else PAGE_SPECS[index].title)


def _scrollable_page(page: QWidget) -> QScrollArea:
    page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(page)
    return scroll


def _page_host(page: QWidget, scroll_policy: str) -> QWidget:
    if scroll_policy == "workbench":
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return page
    return _scrollable_page(page)
