import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
import anomaly_inspection.desktop.runtime as runtime_module
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QGroupBox,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableView,
    QWidget,
)

from anomaly_inspection.core.config import load_config
from anomaly_inspection.core.zone_io import load_zones
from anomaly_inspection.desktop.config_io import config_to_yaml_data, default_config_data, save_config_data, validate_config_data
from anomaly_inspection.desktop.ui.formatting import (
    format_anomaly_decision,
    format_duration_ms,
    format_log_value,
    format_pixel_area,
    format_presence_status,
    format_ratio_percent,
    format_score,
    format_threshold,
)
from anomaly_inspection.desktop.ui.icons import app_logo_path, app_window_icon, qtawesome_available, state_icon
from anomaly_inspection.desktop.ui.layout_contracts import ActionButtonGrid, ActionSection, ControlRail, ElidedValueLabel, WorkbenchLayout
from anomaly_inspection.desktop.pages.inspect_camera import (
    InspectCameraPage,
    captured_image_path,
    default_camera_output_dir,
    save_captured_inspection_frame,
)
from anomaly_inspection.desktop.camera_controller import CameraController, CameraControllerState, CameraStartupPolicy
from anomaly_inspection.desktop.main_window import MainWindow
from anomaly_inspection.desktop.pages.inspect_image import (
    InspectionWorker,
    InspectImagePage,
    config_for_gui_inspection,
    default_inspect_output_dir,
    require_saved_config_path,
)
from anomaly_inspection.desktop.job_paths import (
    default_camera_output_dir as derived_camera_output_dir,
    default_job_asset_root,
    default_image_output_dir,
    default_logs_root_dir,
    default_reference_image_path as job_reference_image_path,
    default_zones_path as job_zones_path,
    make_job_slug,
)
from anomaly_inspection.desktop.pages.logs import (
    LOG_MODE_KEY,
    LOG_SOURCE_KEY,
    LogLoadWorker,
    LogsPage,
    default_logs_output_dir,
    discover_job_history_logs,
    filter_log_rows,
    find_log_files,
    infer_log_mode,
    load_job_history_rows,
    load_inspection_log,
    mode_label,
    row_image_name,
    rows_with_log_metadata,
)
from anomaly_inspection.desktop.pages.project_setup import ProjectSetupPage, RuntimePrepareWorker
from anomaly_inspection.desktop.pages.reference_capture import (
    AspectImageLabel,
    CameraWorker,
    ReferenceCapturePage,
    camera_backend_candidates,
    default_reference_output_path,
    normalize_reference_output_path,
    open_camera_capture,
    save_captured_reference_frame,
)
from anomaly_inspection.desktop.runtime import (
    RUNTIME_NEEDS_REFRESH,
    RUNTIME_NOT_PREPARED,
    RUNTIME_READY,
    PreparedRuntimeManager,
)
from anomaly_inspection.desktop.state import AppState
from anomaly_inspection.desktop.ui.theme import (
    DEFAULT_THEME_NAME,
    apply_app_theme,
    available_theme_names,
    build_app_stylesheet,
    preview_surface_stylesheet,
    resolve_theme,
    table_viewport_stylesheet,
    theme_dimensions,
)
from anomaly_inspection.desktop.ui.components import (
    ImagePreviewWidget,
    MetricGrid,
    PageHeader,
    PathPickerRow,
    ResultBadge,
    ResultImageTabs,
    ResultSummaryPanel,
    ScrollablePane,
    SectionPanel,
    StatusBanner,
)
from anomaly_inspection.desktop.ui.locale import APP_LOCALE, configure_app_locale
from anomaly_inspection.resources import image_asset_path, ui_asset_path
from anomaly_inspection.desktop.pages.zone_editor import (
    ZoneEditorPage,
    compute_zone_canvas_style,
    default_zones_output_path,
    normalize_zones_output_path,
    prepare_polygons_for_save,
    save_zone_session,
)
from anomaly_inspection.core.result_types import FinalResult, InspectionResult


def has_ancestor(widget: QWidget, ancestor_type: type[QWidget], object_name: str | None = None) -> bool:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, ancestor_type) and (object_name is None or parent.objectName() == object_name):
            return True
        parent = parent.parentWidget()
    return False


def test_app_state_defaults():
    state = AppState()

    assert state.config_path is None
    assert state.output_dir is None
    assert state.image_output_dir is None
    assert state.camera_output_dir is None
    assert state.logs_root_dir is None
    assert state.reference_image_path is None
    assert state.zones_path is None
    assert state.reference_path_auto is True
    assert state.zones_path_auto is True
    assert state.inspection_job_name == "Default Job"
    assert state.inspection_job_slug == "default_job"
    assert state.status_message == "Ready"
    assert state.config_loaded is False
    assert state.last_validation_message is None


def test_app_state_accepts_paths():
    state = AppState(
        config_path=Path("configs/local_inspection.yaml"),
        output_dir=Path("outputs/run_001"),
        image_output_dir=Path("outputs/image"),
        camera_output_dir=Path("outputs/camera"),
        logs_root_dir=Path("outputs/logs"),
        reference_image_path=Path("data/reference/empty_reference.png"),
        zones_path=Path("configs/zones.json"),
        inspection_job_name="Metal Surface",
        status_message="Loading",
        config_loaded=True,
        last_validation_message="Valid",
    )

    assert state.config_path == Path("configs/local_inspection.yaml")
    assert state.output_dir == Path("outputs/run_001")
    assert state.image_output_dir == Path("outputs/image")
    assert state.camera_output_dir == Path("outputs/camera")
    assert state.logs_root_dir == Path("outputs/logs")
    assert state.reference_image_path == Path("data/reference/empty_reference.png")
    assert state.zones_path == Path("configs/zones.json")
    assert state.reference_path_auto is False
    assert state.zones_path_auto is False
    assert state.inspection_job_name == "Metal Surface"
    assert state.inspection_job_slug == "metal_surface"
    assert state.status_message == "Loading"
    assert state.config_loaded is True
    assert state.last_validation_message == "Valid"


def test_public_app_modules_import():
    import anomaly_inspection.desktop.main
    import anomaly_inspection.desktop.ui.formatting
    import anomaly_inspection.desktop.ui.icons
    import anomaly_inspection.desktop.job_paths
    import anomaly_inspection.desktop.main_window
    import anomaly_inspection.desktop.pages.reference_capture
    import anomaly_inspection.desktop.pages.inspect_image
    import anomaly_inspection.desktop.pages.inspect_camera
    import anomaly_inspection.desktop.pages.logs
    import anomaly_inspection.desktop.pages.zone_editor
    import anomaly_inspection.desktop.ui.theme
    import anomaly_inspection.desktop.ui.components

    assert callable(anomaly_inspection.desktop.main.main)
    assert anomaly_inspection.desktop.ui.formatting.format_duration_ms(15) == "15 ms"
    assert anomaly_inspection.desktop.ui.icons.app_logo_path().name == "logo.png"
    assert anomaly_inspection.desktop.job_paths.make_job_slug("Transistor") == "transistor"
    assert anomaly_inspection.desktop.main_window.PAGE_SPECS
    assert anomaly_inspection.desktop.pages.reference_capture.ReferenceCapturePage
    assert anomaly_inspection.desktop.pages.inspect_image.InspectImagePage
    assert anomaly_inspection.desktop.pages.inspect_camera.InspectCameraPage
    assert anomaly_inspection.desktop.pages.logs.LogsPage
    assert anomaly_inspection.desktop.pages.zone_editor.ZoneEditorPage
    assert anomaly_inspection.desktop.ui.theme.build_app_stylesheet
    assert anomaly_inspection.desktop.ui.components.PageHeader


def test_theme_stylesheet_contains_shell_rules():
    stylesheet = build_app_stylesheet()

    assert stylesheet
    assert "QMainWindow" in stylesheet
    assert "QFrame#topNavShell" in stylesheet
    assert "QWidget#topPageContextArea" in stylesheet
    assert "QWidget#sectionPanel" in stylesheet
    assert "QFrame#statusBanner" in stylesheet
    assert "QPushButton[buttonRole=\"primary\"]" in stylesheet
    assert "QPushButton[buttonRole=\"secondary\"]:hover" in stylesheet
    assert "QPushButton[buttonRole=\"secondary\"]:disabled:hover" in stylesheet
    assert "QWidget#metricGrid" in stylesheet
    assert "QFrame#resultBadge" in stylesheet
    assert "QGraphicsView#imagePreviewWidget" in stylesheet
    assert "QPushButton#fitImageButton" in stylesheet
    assert "QLineEdit:hover" in stylesheet
    assert "QSpinBox::up-button" in stylesheet
    assert "QSpinBox::down-button" in stylesheet
    assert "QDoubleSpinBox::up-button" in stylesheet
    assert "QDoubleSpinBox::down-button" in stylesheet
    assert "subcontrol-position: top right" in stylesheet
    assert "subcontrol-position: bottom right" in stylesheet
    assert "QSpinBox::up-arrow" in stylesheet
    assert "QSpinBox::down-arrow" in stylesheet
    assert "QDoubleSpinBox::up-arrow" in stylesheet
    assert "QDoubleSpinBox::down-arrow" in stylesheet
    assert "resources/ui/spinbox_arrow_up_on_light.svg" in stylesheet
    assert "resources/ui/spinbox_arrow_down_on_light.svg" in stylesheet
    assert "resources/ui/spinbox_arrow_up_on_dark.svg" in build_app_stylesheet(resolve_theme("factory_dark"))
    assert "resources/ui/spinbox_arrow_down_on_dark.svg" in build_app_stylesheet(resolve_theme("factory_dark"))
    assert "QPushButton:focus" in stylesheet
    assert "QPushButton:disabled:hover" in stylesheet
    assert "QTabBar::tab:selected" in stylesheet
    assert "QTabBar::tab:hover" in stylesheet
    assert "QCheckBox::indicator" in stylesheet
    assert "QCheckBox::indicator:hover" in stylesheet
    assert "QCheckBox::indicator:checked" in stylesheet
    assert "QCheckBox::indicator:disabled" in stylesheet
    assert "QRadioButton::indicator" in stylesheet
    assert "resources/ui/checkbox_check.svg" in stylesheet
    assert "resources/ui/radio_dot.svg" in stylesheet
    assert "QStatusBar" not in stylesheet
    assert DEFAULT_THEME_NAME == "industrial_light_devcpp"
    assert "#f3f3f3" in stylesheet
    assert "#111820" in build_app_stylesheet(resolve_theme("factory_dark"))


def test_icon_helpers_resolve_app_logo_and_semantic_icons():
    app = QApplication.instance() or QApplication([])
    assert app_logo_path().exists()
    assert app_window_icon().isNull() is False
    assert qtawesome_available() is True
    icon = state_icon("ready")
    assert icon is not None


def test_packaged_ui_assets_exist_and_are_referenced_by_the_stylesheet():
    stylesheets = build_app_stylesheet() + build_app_stylesheet(resolve_theme("factory_dark"))
    asset_names = (
        "checkbox_check.svg",
        "checkbox_check_disabled.svg",
        "radio_dot.svg",
        "radio_dot_disabled.svg",
        "spinbox_arrow_disabled.svg",
        "spinbox_arrow_down_on_dark.svg",
        "spinbox_arrow_down_on_light.svg",
        "spinbox_arrow_up_on_dark.svg",
        "spinbox_arrow_up_on_light.svg",
    )

    assert image_asset_path("logo.png").is_file()
    for asset_name in asset_names:
        asset_path = ui_asset_path(asset_name)
        assert asset_path.is_file()
        assert asset_path.as_posix() in stylesheets


def test_theme_registry_contains_factory_presets():
    assert available_theme_names() == (
        "factory_dark",
        "industrial_light_devcpp",
        "high_contrast_factory_light",
    )
    assert resolve_theme().name == "industrial_light_devcpp"
    assert resolve_theme("factory_dark").name == "factory_dark"
    assert resolve_theme("high_contrast_factory_light").name == "high_contrast_factory_light"


def test_page_header_constructs_with_title_and_subtitle():
    app = QApplication.instance() or QApplication([])
    header = PageHeader("Logs", "Review CSV inspection records.")

    labels = header.findChildren(QLabel)

    assert labels[0].text() == "Logs"
    assert labels[0].objectName() == "pageHeaderTitle"
    assert labels[1].text() == "Review CSV inspection records."
    assert labels[1].objectName() == "pageHeaderSubtitle"


def test_section_panel_constructs_with_content_layout():
    app = QApplication.instance() or QApplication([])
    panel = SectionPanel("Model", "Choose model options.")

    labels = panel.findChildren(QLabel)

    assert panel.objectName() == "sectionPanel"
    assert labels[0].text() == "Model"
    assert panel.content_layout is not None


def test_compact_section_panel_marks_density():
    app = QApplication.instance() or QApplication([])
    panel = SectionPanel("Source", compact=True)

    assert panel.property("density") == "compact"
    assert panel.content_layout.spacing() <= 5


def test_classic_section_panel_marks_native_variant():
    app = QApplication.instance() or QApplication([])
    panel = SectionPanel("Source", compact=True, classic=True)

    assert panel.property("density") == "compact"
    assert panel.property("styleVariant") == "classic"
    assert 'QWidget#sectionPanel[styleVariant="classic"]' in build_app_stylesheet()


def test_status_banner_sets_and_clears_message():
    app = QApplication.instance() or QApplication([])
    banner = StatusBanner()

    banner.set_message("Config valid.", "success")

    assert banner.text() == "Config valid."
    assert banner.property("level") == "success"
    assert banner.isVisible() is True

    banner.clear()

    assert banner.text() == ""
    assert banner.isHidden() is True


def test_status_banner_state_supports_icons_and_processing():
    app = QApplication.instance() or QApplication([])
    banner = StatusBanner()

    banner.set_state("Inspecting", "Running inspection", "processing")

    assert banner.property("state") == "processing"
    assert banner.property("level") == "info"
    assert "Inspecting: Running inspection" in banner.text()
    assert banner._spinner_timer.isActive() is True

    banner.set_state("Camera live", "Capture frame", "live")

    assert banner.property("state") == "live"
    assert banner._spinner_timer.isActive() is True

    banner.set_message("Config valid.", "success")

    assert banner.property("state") == ""
    assert banner._spinner_timer.isActive() is False


def test_path_picker_row_gets_and_sets_path_text():
    app = QApplication.instance() or QApplication([])
    row = PathPickerRow("Browse...")

    row.setText("configs/local.yaml")

    assert row.text() == "configs/local.yaml"
    assert row.line_edit.text() == "configs/local.yaml"
    assert row.button.text() == "Browse..."
    assert row.line_edit.minimumWidth() == 0


def test_path_picker_row_exposes_accessible_label_and_buddy():
    app = QApplication.instance() or QApplication([])
    row = PathPickerRow("Browse...", label_text="&Config path")
    label = row.findChild(QLabel, "pathPickerLabel")

    assert label is not None
    assert label.buddy() is row.line_edit
    assert row.line_edit.accessibleName() == "Config path"
    assert "Config path" in row.button.accessibleDescription()


def test_job_slug_sanitizer_normalizes_names():
    assert make_job_slug("Transistor") == "transistor"
    assert make_job_slug("Metal Surface") == "metal_surface"
    assert make_job_slug("Metal/Surface: Test") == "metal_surface_test"
    assert make_job_slug("  Part   A  ") == "part_a"
    assert make_job_slug("") == "default_job"
    assert make_job_slug("CON") == "job_con"


def test_job_output_path_helpers():
    assert derived_camera_output_dir("transistor") == Path("outputs/transistor/camera")
    assert default_image_output_dir("transistor") == Path("outputs/transistor/image")
    assert default_logs_root_dir("transistor") == Path("outputs/transistor")
    assert default_job_asset_root("transistor") == Path("data/jobs/transistor")
    assert job_reference_image_path("transistor") == Path("data/jobs/transistor/reference/empty_reference.png")
    assert job_zones_path("transistor") == Path("data/jobs/transistor/zones/zones.json")


def normalized_path_text(text: str) -> str:
    return text.replace("\\", "/")


def test_metric_grid_sets_metric_values():
    app = QApplication.instance() or QApplication([])
    grid = MetricGrid()

    grid.set_metrics([("Presence Status", "PART_PRESENT"), ("Anomaly Score", "0.123456")])

    assert grid.value_text("Presence Status") == "PART_PRESENT"
    assert grid.value_text("Anomaly Score") == "0.123456"


def test_engineering_formatting_helpers_present_ui_values():
    assert format_duration_ms(43234.3253463) == "43.2 s"
    assert format_duration_ms(15.2) == "15 ms"
    assert format_duration_ms(1234.0) == "1.23 s"
    assert format_score("0.500000") == "0.5"
    assert format_threshold(0.123456) == "0.123"
    assert format_ratio_percent("0.250000") == "25.0%"
    assert format_pixel_area("550.00") == "550 px^2"
    assert format_presence_status("PART_PRESENT") == "Part present"
    assert format_anomaly_decision("False") == "No anomaly"
    assert format_log_value("total_time_ms", "43234.3253463") == "43.2 s"


def test_result_summary_panel_hides_and_shows_error_message():
    app = QApplication.instance() or QApplication([])
    panel = ResultSummaryPanel()

    panel.set_result(
        "NG",
        [("Presence", "Part present"), ("Mean pixel difference", "12.0")],
        csv_log_path="outputs/inspection_log.csv",
    )

    assert isinstance(panel.result_badge, ResultBadge)
    assert panel.result_badge.text() == "NG"
    assert panel.metric_grid.value_text("Presence") == "Part present"
    assert panel.essential_metric_grid.value_text("Presence") == "Part present"
    assert panel.technical_metric_grid.value_text("Mean pixel difference") == "12.0"
    assert panel.csv_text.toPlainText() == "outputs/inspection_log.csv"
    assert panel.error_label.isHidden() is True
    assert panel.error_text.isHidden() is True
    assert panel.technical_button.isHidden() is False
    assert panel.technical_details_widget.isHidden() is True

    panel.technical_button.setChecked(True)
    assert panel.technical_details_widget.isHidden() is False

    panel.set_result("ERROR", [], error_message="failure")

    assert panel.error_label.isHidden() is False
    assert panel.error_text.isHidden() is False
    assert panel.error_text.toPlainText() == "failure"


def test_result_summary_panel_operator_result_states_are_semantic():
    app = QApplication.instance() or QApplication([])
    panel = ResultSummaryPanel()

    for result_name in ("OK", "NG", "NO_PART", "ERROR"):
        panel.set_result(result_name, [])

        expected = {
            "OK": "OK",
            "NG": "NG",
            "NO_PART": "NO PART",
            "ERROR": "ERROR",
        }[result_name]
        assert panel.result_badge.text() == expected
        assert panel.result_badge.property("result") == result_name

    panel.clear()

    assert panel.result_badge.text() == "No result"
    assert panel.result_badge.property("result") == "neutral"


def test_result_summary_panel_hides_technical_section_when_unneeded():
    app = QApplication.instance() or QApplication([])
    panel = ResultSummaryPanel()

    panel.set_result("OK", [("Presence", "Part present")])

    assert panel.technical_button.isHidden() is True
    assert panel.technical_details_widget.isHidden() is True
    assert panel.metric_grid.value_text("Presence") == "Part present"


def test_result_image_tabs_constructs_and_clears():
    app = QApplication.instance() or QApplication([])
    tabs = ResultImageTabs()

    assert tabs.tabs.count() == 3
    assert tabs.fit_image_button.text() == "Fit Image"
    assert tabs.fit_image_button.objectName() == "fitImageButton"
    assert tabs.tabs.cornerWidget(Qt.Corner.TopRightCorner) is tabs.fit_image_button

    tabs.clear()

    assert tabs.annotated_preview._pixmap_item is None
    assert tabs.heatmap_preview._pixmap_item is None
    assert tabs.mask_preview._pixmap_item is None


def test_result_image_tabs_accepts_context_specific_minimum_heights():
    app = QApplication.instance() or QApplication([])
    tabs = ResultImageTabs(min_height=500, tabs_min_height=470, preview_min_height=420)

    assert tabs.minimumHeight() == 500
    assert tabs.tabs.minimumHeight() == 470
    assert tabs.annotated_preview.minimumHeight() == 420
    assert tabs.heatmap_preview.minimumHeight() == 420
    assert tabs.mask_preview.minimumHeight() == 420


def test_scrollable_workflow_rail_uses_comfortable_theme_widths():
    app = QApplication.instance() or QApplication([])
    pane = ScrollablePane(QWidget(), object_name="testWorkflowRail")
    dimensions = theme_dimensions()

    assert dimensions.workflow_rail_size_hint_width >= 540
    assert dimensions.workflow_rail_min_width >= 420
    assert dimensions.inspect_splitter_left_width >= dimensions.workflow_rail_size_hint_width
    assert dimensions.camera_splitter_left_width >= dimensions.workflow_rail_size_hint_width
    assert dimensions.inspect_splitter_right_width > dimensions.inspect_splitter_left_width
    assert dimensions.camera_splitter_right_width > dimensions.camera_splitter_left_width
    assert dimensions.camera_splitter_left_width >= 900
    assert dimensions.camera_splitter_right_width >= 1000
    assert dimensions.camera_splitter_left_stretch == 3
    assert dimensions.camera_splitter_right_stretch == 4
    assert pane.sizeHint().width() == dimensions.workflow_rail_size_hint_width
    assert pane.minimumSizeHint().width() == dimensions.workflow_rail_min_width


def test_elided_value_label_preserves_full_text_in_tooltip():
    app = QApplication.instance() or QApplication([])
    text = "D:/very/long/path/to/model/exported/openvino/model.xml"
    label = ElidedValueLabel(text)

    assert label.full_text() == text
    assert label.toolTip() == text
    assert label.wordWrap() is False
    assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored

    label.set_full_text("short")
    assert label.full_text() == "short"
    assert label.toolTip() == "short"


def test_action_section_has_fixed_vertical_contract():
    app = QApplication.instance() or QApplication([])
    section = ActionSection(rows=2)
    first = QPushButton("First")
    second = QPushButton("Second")
    first_row = section.add_row((first,))
    second_row = section.add_row((second,))

    assert section.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert first.minimumHeight() == theme_dimensions().button_min_height
    assert second.minimumHeight() == theme_dimensions().button_min_height
    assert section.minimumHeight() >= first_row.sizeHint().height() + second_row.sizeHint().height()
    assert section.maximumHeight() > section.minimumHeight()
    assert not has_ancestor(section, QScrollArea)


def test_action_section_reserves_real_button_size_hint_height():
    app = QApplication.instance() or QApplication([])
    section = ActionSection(rows=1)
    button = QPushButton("Run Inspection")
    row = section.add_row((button,), align_right=True)

    assert button.minimumHeight() == theme_dimensions().button_min_height
    assert row.sizeHint().height() >= button.sizeHint().height()
    assert section.minimumHeight() >= row.sizeHint().height()


def test_control_rail_bounds_width_and_keeps_fixed_widgets_outside_scroll():
    app = QApplication.instance() or QApplication([])
    rail = ControlRail(object_name="testControlRail")
    fixed = QPushButton("Run")
    secondary = QLabel("Secondary")

    rail.add_fixed(fixed)
    scroll = rail.set_scroll_body(secondary, object_name="testControlRailScroll")

    assert rail.minimumWidth() == theme_dimensions().workflow_rail_min_width
    assert rail.maximumWidth() == theme_dimensions().workflow_rail_size_hint_width
    assert rail.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert has_ancestor(fixed, QScrollArea) is False
    assert has_ancestor(secondary, QScrollArea, "testControlRailScroll") is True


def test_workbench_layout_uses_non_collapsible_bounded_rail_and_expanding_workspace():
    app = QApplication.instance() or QApplication([])
    workbench = WorkbenchLayout(object_name="testWorkbench")
    rail = ControlRail(object_name="testRail")
    workspace = QWidget()

    workbench.set_control_rail(rail)
    workbench.set_main_workspace(workspace)

    assert workbench.childrenCollapsible() is False
    assert workbench.count() == 2
    assert workbench.widget(0) is rail
    assert workbench.widget(1) is workspace
    assert rail.maximumWidth() == theme_dimensions().workflow_rail_size_hint_width
    assert workspace.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_main_window_keeps_pages_in_fixed_content_hosts():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    first_page_host = window._stack.widget(0)

    assert first_page_host is window._project_setup_page
    assert not isinstance(first_page_host, QScrollArea)


def test_main_window_keeps_every_workflow_page_in_a_fixed_host():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    assert window._stack.widget(0) is window._project_setup_page
    assert window._stack.widget(1) is window._reference_capture_page
    assert window._stack.widget(2) is window._zone_editor_page
    assert window._stack.widget(3) is window._inspect_image_page
    assert not isinstance(window._stack.widget(3), QScrollArea)
    assert window._stack.widget(4) is window._inspect_camera_page
    assert not isinstance(window._stack.widget(4), QScrollArea)
    assert window._stack.widget(5) is window._logs_page



def test_main_window_top_navigation_contains_workflow_group_labels_and_buttons():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    labels = {label.text() for label in window.findChildren(QLabel) if label.objectName() == "navGroupLabel"}
    buttons = {button.text() for button in window.findChildren(QPushButton) if button.objectName() == "navButton"}
    theme_combo = window.findChild(QComboBox, "themeSelector")

    assert window.findChild(QWidget, "topPageContextArea") is not None
    assert window.findChild(QWidget, "topBrandArea") is not None
    assert window.findChild(QWidget, "topNavCluster") is not None
    assert window.findChild(QWidget, "topThemeArea") is not None
    assert window.page_context_title.text() == "Project Setup"
    assert window.page_context_subtitle.text() == "Prepare the inspection job and required files."
    assert {"Setup", "Inspection", "Review"}.issubset(labels)
    assert "Theme" in labels
    assert theme_combo is not None
    assert {theme_combo.itemData(index) for index in range(theme_combo.count())} == set(available_theme_names())
    assert buttons == {
        "Project Setup",
        "Capture Reference",
        "Draw Zones",
        "Inspect Image",
        "Inspect Camera",
        "Logs",
    }


def test_main_window_top_navigation_compacts_at_minimum_width():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    window.resize(960, 640)
    window._update_shell_compact_mode(960)
    app.processEvents()

    buttons = {button.text() for button in window.findChildren(QPushButton) if button.objectName() == "navButton"}
    theme_combo = window.findChild(QComboBox, "themeSelector")

    assert buttons == {"Setup", "Reference", "Zones", "Image", "Camera", "Logs"}
    assert theme_combo is not None
    assert theme_combo.isHidden() is False
    assert window.page_context_subtitle.isHidden() is True
    assert window.findChild(QWidget, "topBrandArea").isHidden() is True


def test_main_window_theme_selector_reapplies_custom_theme_surfaces():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    window = MainWindow(state)
    combo = window.findChild(QComboBox, "themeSelector")
    assert combo is not None

    try:
        dark_index = combo.findData("factory_dark")
        assert dark_index >= 0
        combo.setCurrentIndex(dark_index)

        assert state.desktop_theme_name == "factory_dark"
        assert window._logs_page is not None
        assert window._reference_capture_page is not None
        assert window._zone_editor_page is not None
        assert window._logs_page.records_table.viewport().styleSheet() == table_viewport_stylesheet()
        assert window._reference_capture_page.preview_label.styleSheet() == preview_surface_stylesheet()
        assert window._zone_editor_page.view.styleSheet() == preview_surface_stylesheet()
    finally:
        light_index = combo.findData(DEFAULT_THEME_NAME)
        if light_index >= 0:
            combo.setCurrentIndex(light_index)
        apply_app_theme(app, DEFAULT_THEME_NAME)


def test_main_window_top_navigation_switches_pages_and_active_button():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    logs_button = next(
        button for button in window.findChildren(QPushButton) if button.objectName() == "navButton" and button.text() == "Logs"
    )
    logs_button.click()

    assert window._stack.currentIndex() == 5
    assert logs_button.isChecked() is True
    assert window.page_context_title.text() == "Logs"
    assert window.page_context_subtitle.text() == "Review inspection history and artifacts."


def test_main_window_does_not_create_global_status_bar():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    assert window.findChildren(QStatusBar) == []


def test_main_window_defers_close_until_active_work_finishes():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    class ShutdownStub:
        def __init__(self, *results):
            self.results = list(results)

        def shutdown(self):
            if len(self.results) > 1:
                return self.results.pop(0)
            return self.results[0]

    window._reference_capture_page = ShutdownStub(True)
    window._project_setup_page = ShutdownStub(False, True)
    window._inspect_image_page = ShutdownStub(True)
    window._inspect_camera_page = ShutdownStub(True)
    window._logs_page = ShutdownStub(True)

    first_event = QCloseEvent()
    window.closeEvent(first_event)

    assert first_event.isAccepted() is False
    assert window._close_pending is True
    assert window._close_retry_timer.isActive() is True

    window._close_retry_timer.stop()
    assert window._project_setup_page.shutdown() is True


def test_desktop_pages_do_not_repeat_body_page_headers():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())

    for index in range(window._stack.count()):
        host = window._stack.widget(index)
        page = host.widget() if isinstance(host, QScrollArea) else host
        assert not page.findChildren(QLabel, "pageHeaderTitle")
        assert not page.findChildren(QLabel, "pageHeaderSubtitle")


def test_config_io_saves_backend_compatible_yaml(tmp_path):
    data = default_config_data()
    data["project"]["name"] = "Metal Surface"
    data["model"]["path"] = "model.pt"
    data["model"]["anomalib_model"] = "reverse_distillation"
    data["presence"]["reference_image_path"] = "data/reference/empty_reference.png"
    data["presence"]["zones_path"] = "configs/zones.json"
    data["output"]["show_images"] = True
    path = tmp_path / "local_inspection.yaml"

    save_config_data(path, data)

    loaded = load_config(path)
    assert loaded.project.name == "Metal Surface"
    assert loaded.model.path == tmp_path / "model.pt"
    assert loaded.model.anomalib_model == "reverse_distillation"
    assert loaded.model.checkpoint_inference_mode == "engine"
    assert loaded.presence.pixel_diff_threshold == 30
    assert loaded.output.show_images is True

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(raw) == ["project", "model", "presence", "output"]
    assert raw["project"]["name"] == "Metal Surface"
    assert raw["model"]["anomalib_model"] == "reverse_distillation"
    assert "checkpoint_inference_mode" not in raw["model"]


def test_config_io_serialization_omits_checkpoint_inference_mode(tmp_path):
    data = default_config_data()
    data["model"]["path"] = "model.ckpt"
    data["model"]["format"] = "ckpt"
    data["presence"]["reference_image_path"] = "data/reference/empty_reference.png"
    data["presence"]["zones_path"] = "configs/zones.json"
    path = tmp_path / "inspection.yaml"

    save_config_data(path, data)
    loaded = load_config(path)
    raw = config_to_yaml_data(loaded)

    assert loaded.model.checkpoint_inference_mode == "engine"
    assert "checkpoint_inference_mode" not in raw["model"]


def test_config_io_drops_deprecated_direct_checkpoint_inference_mode(tmp_path):
    data = default_config_data()
    data["model"]["path"] = "model.ckpt"
    data["model"]["format"] = "ckpt"
    data["model"]["checkpoint_inference_mode"] = "direct"
    data["presence"]["reference_image_path"] = "data/reference/empty_reference.png"
    data["presence"]["zones_path"] = "configs/zones.json"
    path = tmp_path / "inspection.yaml"

    save_config_data(path, data)
    with pytest.warns(DeprecationWarning, match="checkpoint_inference_mode is deprecated"):
        loaded = load_config(path)
    raw = config_to_yaml_data(loaded)

    assert loaded.model.checkpoint_inference_mode == "direct"
    assert "checkpoint_inference_mode" not in raw["model"]


def test_config_io_validation_uses_backend_rules():
    data = default_config_data()
    data["model"]["path"] = "model.pt"
    data["presence"]["blur_kernel_size"] = 4

    try:
        validate_config_data(data)
    except ValueError as exc:
        assert "presence.blur_kernel_size" in str(exc)
    else:
        raise AssertionError("Expected backend validation to reject even blur kernel size.")


def test_config_io_validation_rejects_blank_required_paths():
    data = default_config_data()
    data["presence"]["reference_image_path"] = ""

    try:
        validate_config_data(data)
    except ValueError as exc:
        assert "presence.reference_image_path is required" in str(exc)
    else:
        raise AssertionError("Expected validation to reject blank reference image path.")


def valid_existing_file_config(tmp_path):
    model_path = tmp_path / "model.pt"
    reference_path = tmp_path / "empty_reference.png"
    zones_path = tmp_path / "zones.json"
    model_path.write_bytes(b"placeholder model")
    reference_path.write_bytes(b"placeholder image")
    zones_path.write_text("{}", encoding="utf-8")

    data = default_config_data()
    data["model"]["path"] = str(model_path)
    data["presence"]["reference_image_path"] = str(reference_path)
    data["presence"]["zones_path"] = str(zones_path)
    return data, model_path, reference_path, zones_path


def test_config_io_file_existence_validation_passes_when_required_files_exist(tmp_path):
    data, _, _, _ = valid_existing_file_config(tmp_path)

    config = validate_config_data(data, tmp_path / "local_inspection.yaml", require_existing_files=True)

    assert config.model.path.exists()
    assert config.presence.reference_image_path.exists()
    assert config.presence.zones_path.exists()


def test_config_io_file_existence_validation_fails_when_model_missing(tmp_path):
    data, model_path, _, _ = valid_existing_file_config(tmp_path)
    model_path.unlink()

    try:
        validate_config_data(data, tmp_path / "local_inspection.yaml", require_existing_files=True)
    except ValueError as exc:
        assert f"Model file does not exist: {model_path}" in str(exc)
    else:
        raise AssertionError("Expected validation to reject missing model file.")


def test_config_io_file_existence_validation_fails_when_reference_missing(tmp_path):
    data, _, reference_path, _ = valid_existing_file_config(tmp_path)
    reference_path.unlink()

    try:
        validate_config_data(data, tmp_path / "local_inspection.yaml", require_existing_files=True)
    except ValueError as exc:
        assert f"Reference image does not exist: {reference_path}" in str(exc)
    else:
        raise AssertionError("Expected validation to reject missing reference image.")


def test_config_io_file_existence_validation_fails_when_zones_missing(tmp_path):
    data, _, _, zones_path = valid_existing_file_config(tmp_path)
    zones_path.unlink()

    try:
        validate_config_data(data, tmp_path / "local_inspection.yaml", require_existing_files=True)
    except ValueError as exc:
        assert f"Zones JSON does not exist: {zones_path}" in str(exc)
    else:
        raise AssertionError("Expected validation to reject missing zones JSON.")


def test_config_io_file_existence_validation_reports_multiple_missing_files(tmp_path):
    data, model_path, reference_path, zones_path = valid_existing_file_config(tmp_path)
    model_path.unlink()
    reference_path.unlink()
    zones_path.unlink()

    try:
        validate_config_data(data, tmp_path / "local_inspection.yaml", require_existing_files=True)
    except ValueError as exc:
        message = str(exc)
        assert f"Model file does not exist: {model_path}" in message
        assert f"Reference image does not exist: {reference_path}" in message
        assert f"Zones JSON does not exist: {zones_path}" in message
    else:
        raise AssertionError("Expected validation to report all missing required files.")


def test_prepared_runtime_manager_prepares_and_invalidates(tmp_path, monkeypatch):
    data, _, _, _ = valid_existing_file_config(tmp_path)
    config_path = tmp_path / "local_inspection.yaml"
    save_config_data(config_path, data)

    class FakePipeline:
        def __init__(self, config):
            self.config = config
            self.prepared = False

        def prepare_anomaly_backend(self):
            self.prepared = True

        def inspect_image(self, image_path, output_dir, inspection_mode="image"):
            return InspectionResult(image_path=str(image_path), final_result=FinalResult.OK, inspection_mode=inspection_mode)

    monkeypatch.setattr(runtime_module, "InspectionPipeline", FakePipeline)
    manager = PreparedRuntimeManager()

    assert manager.state == RUNTIME_NOT_PREPARED

    manager.prepare(config_path)

    assert manager.state == RUNTIME_READY
    assert manager.is_current(config_path) is True

    manager.mark_stale()

    assert manager.state == RUNTIME_NEEDS_REFRESH
    assert manager.is_current(config_path) is False


def test_inspection_worker_uses_prepared_runtime_when_available(tmp_path):
    config_path = tmp_path / "config.yaml"
    image_path = tmp_path / "part.png"
    output_dir = tmp_path / "out"
    config_path.write_text("model: {}\n", encoding="utf-8")
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    completed = []

    class FakeRuntimeManager:
        def is_current(self, path):
            return True

        def inspect_image(self, config_path, image_path, output_dir, *, inspection_mode="image"):
            return (
                InspectionResult(
                    image_path=str(image_path),
                    final_result=FinalResult.OK,
                    inspection_mode=inspection_mode,
                ),
                True,
            )

    worker = InspectionWorker(config_path, image_path, output_dir, runtime_manager=FakeRuntimeManager())
    worker.completed.connect(lambda result, path: completed.append((result, path)))

    worker.run()

    assert len(completed) == 1
    assert completed[0][0].final_result == FinalResult.OK
    assert completed[0][0].inspection_mode == "image"


def test_inspection_worker_cancel_before_start_discards_work(tmp_path):
    worker = InspectionWorker(tmp_path / "config.yaml", tmp_path / "image.png", tmp_path / "output")
    cancelled = []
    completed = []
    failed = []
    finished = []
    worker.cancelled.connect(cancelled.append)
    worker.completed.connect(lambda *args: completed.append(args))
    worker.failed.connect(failed.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.request_cancel()
    worker.run()

    assert cancelled == ["Inspection cancelled before it started."]
    assert completed == []
    assert failed == []
    assert finished == [True]
    assert (tmp_path / "output").exists() is False


def test_runtime_prepare_worker_cancel_before_start_does_not_prepare(tmp_path):
    class FakeRuntimeManager:
        def __init__(self):
            self.prepare_calls = []

        def prepare(self, path):
            self.prepare_calls.append(path)

        def clear(self):
            raise AssertionError("clear should not be needed before preparation starts")

    manager = FakeRuntimeManager()
    worker = RuntimePrepareWorker(manager, tmp_path / "config.yaml")
    cancelled = []
    finished = []
    worker.cancelled.connect(cancelled.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.request_cancel()
    worker.run()

    assert manager.prepare_calls == []
    assert cancelled == ["Runtime preparation cancelled before it started."]
    assert finished == [True]


def test_project_setup_page_uses_shared_ui_components():
    app = QApplication.instance() or QApplication([])
    page = ProjectSetupPage(AppState())

    panels = page.findChildren(SectionPanel)

    assert len(panels) == 6
    assert isinstance(page.feedback_label, StatusBanner)
    assert isinstance(page.readiness_banner, StatusBanner)
    assert page.findChild(QWidget, "setupConsole") is not None
    assert page.findChild(QWidget, "setupConsoleColumn") is not None
    assert "Incomplete:" in page.readiness_banner.text()
    assert page.readiness_banner.property("state") == "blocked"
    assert page.readiness_grid.value_text("Config file") == "Not selected"
    assert page.job_name_edit is not None
    assert page.config_path_edit is not None
    buttons_by_text = {button.text(): button for button in page.findChildren(QPushButton)}
    assert buttons_by_text["Save Config"].property("buttonRole") == "secondary"
    assert buttons_by_text["Save Config As..."].property("buttonRole") == "secondary"
    assert buttons_by_text["Validate Config"].property("buttonRole") == "secondary"
    assert buttons_by_text["Prepare Runtime"].property("buttonRole") == "secondary"
    assert isinstance(page.runtime_banner, StatusBanner)
    assert page.runtime_banner.property("density") == "inline"
    assert page.findChild(QWidget, "runtimePreparationRow") is not None
    assert "Not prepared:" in page.runtime_banner.text()
    assert page.model_path_edit is not None
    assert page.anomalib_model_edit is not None
    assert page.reference_path_edit is not None
    assert page.zones_path_edit is not None
    assert page.anomaly_threshold_spin.decimals() == 3
    assert page.foreground_ratio_spin.decimals() == 3
    assert page.presence_tuning_button.isChecked() is False
    assert page.presence_tuning_widget.isHidden() is True

    page.presence_tuning_button.setChecked(True)

    assert page.presence_tuning_widget.isHidden() is False


def test_project_setup_preserves_anomalib_model_field():
    app = QApplication.instance() or QApplication([])
    page = ProjectSetupPage(AppState())
    data = default_config_data()
    data["model"]["anomalib_model"] = "reverse_distillation"

    page.populate_from_data(data)

    assert page.anomalib_model_edit.text() == "reverse_distillation"
    assert page.to_config_data()["model"]["anomalib_model"] == "reverse_distillation"


def test_project_setup_drops_checkpoint_inference_mode_without_ui_field():
    app = QApplication.instance() or QApplication([])
    page = ProjectSetupPage(AppState())
    data = default_config_data()
    data["model"]["checkpoint_inference_mode"] = "direct"

    page.populate_from_data(data)

    assert "checkpoint_inference_mode" not in page.to_config_data()["model"]


def test_project_setup_prepare_runtime_is_primary_only_when_action_is_needed():
    app = QApplication.instance() or QApplication([])
    manager = PreparedRuntimeManager()
    page = ProjectSetupPage(AppState(), runtime_manager=manager)

    assert page.prepare_runtime_button.property("buttonRole") == "primary"

    manager.mark_preparing()
    page._refresh_runtime_status()
    assert page.prepare_runtime_button.property("buttonRole") == "secondary"

    manager.set_error("failed")
    page._refresh_runtime_status()
    assert page.prepare_runtime_button.property("buttonRole") == "secondary"

    manager.mark_stale()
    page._refresh_runtime_status()
    assert page.prepare_runtime_button.property("buttonRole") == "primary"


def test_project_setup_explains_show_images_is_for_cli_runs():
    app = QApplication.instance() or QApplication([])
    page = ProjectSetupPage(AppState())

    assert page.show_images_checkbox.text() == "Show images in CLI runs"
    assert "Desktop inspections" in page.show_images_checkbox.toolTip()
    assert page.show_images_checkbox.accessibleDescription() == page.show_images_checkbox.toolTip()


def test_project_setup_updates_app_state_job_name_and_slug():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = ProjectSetupPage(state)

    page.job_name_edit.setText("Metal/Surface: Test")

    assert state.inspection_job_name == "Metal/Surface: Test"
    assert state.inspection_job_slug == "metal_surface_test"


def test_project_setup_uses_job_centered_setup_asset_defaults():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = ProjectSetupPage(state)

    assert page.reference_path_edit.text() == str(job_reference_image_path("default_job"))
    assert page.zones_path_edit.text() == str(job_zones_path("default_job"))
    assert state.reference_image_path == job_reference_image_path("default_job")
    assert state.zones_path == job_zones_path("default_job")
    assert state.reference_path_auto is True
    assert state.zones_path_auto is True


def test_project_setup_job_change_updates_only_auto_setup_asset_paths():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = ProjectSetupPage(state)

    page.job_name_edit.setText("Transistor")

    assert page.reference_path_edit.text() == str(job_reference_image_path("transistor"))
    assert page.zones_path_edit.text() == str(job_zones_path("transistor"))

    page.reference_path_edit.setText("data/reference/manual.png")
    page.zones_path_edit.setText("configs/custom_zones.json")
    page.job_name_edit.setText("Metal Surface")

    assert page.reference_path_edit.text() == "data/reference/manual.png"
    assert page.zones_path_edit.text() == "configs/custom_zones.json"
    assert state.reference_path_auto is False
    assert state.zones_path_auto is False


def test_project_setup_loaded_legacy_paths_remain_explicit():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = ProjectSetupPage(state)
    data = default_config_data()
    data["presence"]["reference_image_path"] = "data/reference/empty_reference.png"
    data["presence"]["zones_path"] = "configs/zones.json"

    page.populate_from_data(data, explicit_config=True)
    page.job_name_edit.setText("Transistor")

    assert page.reference_path_edit.text() == "data/reference/empty_reference.png"
    assert page.zones_path_edit.text() == "configs/zones.json"
    assert state.reference_path_auto is False
    assert state.zones_path_auto is False


def test_project_setup_readiness_summary_tracks_required_files(tmp_path):
    app = QApplication.instance() or QApplication([])
    config_path = tmp_path / "local_inspection.yaml"
    model_path = tmp_path / "model.pt"
    reference_path = tmp_path / "reference.png"
    zones_path = tmp_path / "zones.json"
    for path in (config_path, model_path, reference_path, zones_path):
        path.write_text("placeholder", encoding="utf-8")
    state = AppState()
    page = ProjectSetupPage(state)

    page._set_config_path(config_path)
    page.model_path_edit.setText(str(model_path))
    page.reference_path_edit.setText(str(reference_path))
    page.zones_path_edit.setText(str(zones_path))
    page._refresh_readiness_summary()

    assert "Ready:" in page.readiness_banner.text()
    assert page.readiness_banner.property("state") == "ready"
    assert page.readiness_grid.value_text("Model artifact").startswith("Available:")
    assert page.readiness_grid.value_text("Reference image").startswith("Available:")
    assert page.readiness_grid.value_text("Zones JSON").startswith("Available:")


def test_project_setup_config_data_uses_displayed_explicit_setup_paths():
    app = QApplication.instance() or QApplication([])
    page = ProjectSetupPage(AppState())

    page.reference_path_edit.setText("data/reference/manual.png")
    page.zones_path_edit.setText("configs/custom_zones.json")
    data = page.to_config_data()

    assert data["presence"]["reference_image_path"] == "data/reference/manual.png"
    assert data["presence"]["zones_path"] == "configs/custom_zones.json"


def test_default_reference_output_path_uses_app_state():
    state = AppState(reference_image_path=Path("custom/reference.png"))

    assert default_reference_output_path(state) == Path("custom/reference.png")


def test_default_reference_output_path_falls_back_to_job_reference():
    assert default_reference_output_path(AppState(inspection_job_name="Transistor")) == job_reference_image_path("transistor")


def test_camera_backend_candidates_prefers_windows_dshow_then_msmf():
    names = [name for name, _ in camera_backend_candidates("win32")]

    assert names == ["DSHOW", "MSMF"]


def test_camera_backend_candidates_uses_default_off_windows():
    assert camera_backend_candidates("linux") == [("DEFAULT", None)]


class FakeCapture:
    def __init__(self, opened: bool, backend_name: str = "FAKE") -> None:
        self._opened = opened
        self._backend_name = backend_name
        self.released = False

    def isOpened(self):
        return self._opened

    def getBackendName(self):
        return self._backend_name

    def release(self):
        self.released = True


class SequenceCapture(FakeCapture):
    def __init__(self, frames: list[np.ndarray], opened: bool = True, backend_name: str = "FAKE") -> None:
        super().__init__(opened, backend_name)
        self.frames = list(frames)
        self.set_calls = []

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True


def test_open_camera_capture_tries_windows_fallback_until_open():
    calls = []
    captures = [FakeCapture(False, "DSHOW"), FakeCapture(True, "MSMF")]

    def factory(index, backend=None):
        calls.append((index, backend))
        return captures[len(calls) - 1]

    capture, backend_name = open_camera_capture(0, factory, "win32")

    assert capture is captures[1]
    assert "MSMF" in backend_name
    assert captures[0].released is True
    assert len(calls) == 2


def test_open_camera_capture_reports_clear_error_when_all_backends_fail():
    captures = [FakeCapture(False, "DSHOW"), FakeCapture(False, "MSMF")]

    def factory(index, backend=None):
        return captures.pop(0)

    try:
        open_camera_capture(0, factory, "win32")
    except RuntimeError as exc:
        assert "Unable to open camera index 0 using DSHOW or MSMF" in str(exc)
    else:
        raise AssertionError("Expected open failure to raise a clear RuntimeError.")


def test_camera_worker_emits_session_payload_after_valid_startup_frame():
    blank = np.zeros((12, 20, 3), dtype=np.uint8)
    valid = np.full((12, 20, 3), 128, dtype=np.uint8)
    capture = SequenceCapture([blank, valid])

    def factory(index, backend=None):
        return capture

    worker = CameraWorker(0, capture_factory=factory, session_id=42)
    frames = []
    resolutions = []
    errors = []
    statuses = []
    finished = []
    worker.frame_ready.connect(lambda session_id, frame: frames.append((session_id, frame.copy())))
    worker.actual_resolution.connect(lambda session_id, width, height: resolutions.append((session_id, width, height)))
    worker.status.connect(lambda session_id, message, level: statuses.append((session_id, message, level)))
    worker.error.connect(lambda session_id, message: errors.append((session_id, message)))
    worker.finished.connect(lambda session_id: finished.append(session_id))

    worker.run()

    assert [(session_id, frame.shape) for session_id, frame in frames] == [(42, valid.shape)]
    assert resolutions == [(42, 20, 12)]
    assert statuses == [(42, "Warming up camera...", "info")]
    assert errors == [(42, "Could not read camera frame.")]
    assert finished == [42]
    assert capture.released is True


def test_camera_worker_recovers_blank_startup_by_reopening_same_backend():
    blank = np.zeros((12, 20, 3), dtype=np.uint8)
    valid = np.full((12, 20, 3), 128, dtype=np.uint8)
    captures = [
        SequenceCapture([blank, blank, blank], backend_name="DSHOW"),
        SequenceCapture([valid], backend_name="DSHOW"),
    ]
    calls = []

    def factory(index, backend=None):
        calls.append((index, backend))
        return captures[len(calls) - 1]

    worker = CameraWorker(
        0,
        capture_factory=factory,
        session_id=7,
        blank_recovery_frames=3,
        blank_final_fail_frames=3,
    )
    frames = []
    statuses = []
    errors = []
    finished = []
    worker.frame_ready.connect(lambda session_id, frame: frames.append((session_id, frame.copy())))
    worker.status.connect(lambda session_id, message, level: statuses.append((session_id, message, level)))
    worker.error.connect(lambda session_id, message: errors.append((session_id, message)))
    worker.finished.connect(lambda session_id: finished.append(session_id))

    worker.run()

    assert [(session_id, frame.shape) for session_id, frame in frames] == [(7, valid.shape)]
    assert errors == [(7, "Could not read camera frame.")]
    assert finished == [7]
    assert (7, "Blank startup frames detected; reopening camera...", "warning") in statuses
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1]
    assert captures[0].released is True
    assert captures[1].released is True


def test_camera_worker_reports_error_after_blank_recovery_fails():
    blank = np.zeros((12, 20, 3), dtype=np.uint8)
    captures = [
        SequenceCapture([blank, blank], backend_name="DSHOW"),
        SequenceCapture([blank, blank], backend_name="DSHOW"),
    ]
    calls = []

    def factory(index, backend=None):
        calls.append((index, backend))
        return captures[len(calls) - 1]

    worker = CameraWorker(
        0,
        capture_factory=factory,
        session_id=9,
        blank_recovery_frames=2,
        blank_final_fail_frames=2,
    )
    frames = []
    errors = []
    finished = []
    worker.frame_ready.connect(lambda session_id, frame: frames.append((session_id, frame.copy())))
    worker.error.connect(lambda session_id, message: errors.append((session_id, message)))
    worker.finished.connect(lambda session_id: finished.append(session_id))

    worker.run()

    assert frames == []
    assert errors == [
        (
            9,
            "Camera opened but frames stayed blank. "
            "Please retry or check exposure, lens, privacy shutter, or driver.",
        )
    ]
    assert finished == [9]
    assert len(calls) == 2
    assert all(capture.released for capture in captures)


def test_camera_worker_stop_during_blank_warmup_cancels_cleanly():
    blank = np.zeros((12, 20, 3), dtype=np.uint8)
    capture = SequenceCapture([blank, blank, blank], backend_name="DSHOW")

    def factory(index, backend=None):
        return capture

    worker = CameraWorker(0, capture_factory=factory, session_id=11, blank_recovery_frames=3)
    frames = []
    errors = []
    finished = []
    worker.frame_ready.connect(lambda session_id, frame: frames.append((session_id, frame.copy())))
    worker.status.connect(lambda session_id, message, level: worker.stop())
    worker.error.connect(lambda session_id, message: errors.append((session_id, message)))
    worker.finished.connect(lambda session_id: finished.append(session_id))

    worker.run()

    assert frames == []
    assert errors == []
    assert finished == [11]
    assert capture.released is True


class FakeControllerWorker(QObject):
    frame_ready = Signal(int, object)
    actual_resolution = Signal(int, int, int)
    status = Signal(int, str, str)
    error = Signal(int, str)
    finished = Signal(int)

    instances: list["FakeControllerWorker"] = []

    def __init__(self, camera_index=0, width=0, height=0, *, session_id=0, **kwargs):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.session_id = session_id
        self.kwargs = kwargs
        self.stop_count = 0
        FakeControllerWorker.instances.append(self)

    @Slot()
    def run(self):
        return

    @Slot()
    def stop(self):
        self.stop_count += 1
        self.finished.emit(self.session_id)


def process_qt_events(app: QApplication, iterations: int = 3) -> None:
    for _ in range(iterations):
        app.processEvents()


def finish_log_operation(page: LogsPage, app: QApplication) -> None:
    thread = page._load_thread
    assert thread is not None
    for _ in range(2000):
        app.processEvents()
        if thread.isFinished():
            break
        QThread.msleep(1)
    assert thread.isFinished() is True
    process_qt_events(app, iterations=10)
    assert page._load_thread is None


def shutdown_camera_controller(controller: CameraController, app: QApplication) -> None:
    worker = controller._worker
    thread = controller._thread
    controller.shutdown()
    if worker is not None:
        worker.finished.emit(controller.session_id())
    process_qt_events(app, iterations=10)
    if thread is not None:
        try:
            thread.quit()
            assert thread.wait(1000) is True
        except RuntimeError:
            pass
    process_qt_events(app, iterations=10)
    assert controller.shutdown() is True


def make_camera_controller(app: QApplication) -> CameraController:
    FakeControllerWorker.instances.clear()
    controller = CameraController(
        lambda: (0, 0, 0),
        startup_policy=CameraStartupPolicy(idle_release_timeout_ms=10),
        worker_factory=FakeControllerWorker,
    )
    return controller


def test_camera_controller_start_creates_one_worker_and_coalesces_repeated_start():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)

    controller.start_preview()
    process_qt_events(app)
    controller.start_preview()

    assert len(FakeControllerWorker.instances) == 1
    assert controller.state() == CameraControllerState.OPENING.value
    assert FakeControllerWorker.instances[0].kwargs["blank_max_value"] == 2
    assert FakeControllerWorker.instances[0].kwargs["blank_mean_max"] == 1.0

    shutdown_camera_controller(controller, app)
    process_qt_events(app)


def test_camera_controller_soft_stop_resume_does_not_reopen_worker():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)
    frames = []
    controller.frame_ready.connect(frames.append)
    frame = np.full((12, 20, 3), 128, dtype=np.uint8)

    controller.start_preview()
    process_qt_events(app)
    session_id = controller.session_id()
    controller._on_worker_frame_ready(session_id, frame)
    controller.stop_preview()
    controller.start_preview()

    assert controller.state() == CameraControllerState.RUNNING.value
    assert len(FakeControllerWorker.instances) == 1
    assert len(frames) == 1

    shutdown_camera_controller(controller, app)
    process_qt_events(app)


def test_camera_controller_idle_release_hard_stops_worker():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)
    frame = np.full((12, 20, 3), 128, dtype=np.uint8)

    controller.start_preview()
    process_qt_events(app)
    worker = FakeControllerWorker.instances[-1]
    thread = controller._thread
    controller._on_worker_frame_ready(controller.session_id(), frame)
    controller.stop_preview()
    controller._release_idle_worker()
    worker.finished.emit(controller.session_id())
    process_qt_events(app, iterations=10)
    if thread is not None:
        try:
            thread.quit()
            assert thread.wait(1000) is True
        except RuntimeError:
            pass

    assert worker.kwargs["stop_event"].is_set() is True
    assert controller.state() == CameraControllerState.RELEASED.value
    assert controller.has_active_worker() is False


def test_camera_controller_stop_during_opening_cancels_worker():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)

    controller.start_preview()
    process_qt_events(app)
    worker = FakeControllerWorker.instances[-1]
    thread = controller._thread
    controller.stop_preview()
    worker.finished.emit(controller.session_id())
    process_qt_events(app, iterations=10)
    if thread is not None:
        try:
            thread.quit()
            assert thread.wait(1000) is True
        except RuntimeError:
            pass

    assert worker.kwargs["stop_event"].is_set() is True
    assert controller.state() == CameraControllerState.RELEASED.value
    assert controller.has_active_worker() is False


def test_camera_controller_start_while_warming_does_not_duplicate_worker():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)

    controller.start_preview()
    process_qt_events(app)
    controller._on_worker_status(controller.session_id(), "Warming up camera...", "info")
    controller.start_preview()

    assert controller.state() == CameraControllerState.WARMING.value
    assert len(FakeControllerWorker.instances) == 1

    shutdown_camera_controller(controller, app)
    process_qt_events(app)


def test_camera_controller_ignores_frames_while_soft_stopped_and_stale_frames():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)
    frames = []
    controller.frame_ready.connect(frames.append)
    frame = np.full((12, 20, 3), 128, dtype=np.uint8)
    stale_frame = np.full((12, 20, 3), 32, dtype=np.uint8)

    controller.start_preview()
    process_qt_events(app)
    session_id = controller.session_id()
    controller._on_worker_frame_ready(session_id, frame)
    controller._on_worker_frame_ready(session_id - 1, stale_frame)
    controller.stop_preview()
    controller._on_worker_frame_ready(session_id, frame)

    assert len(frames) == 1
    assert frames[0] is frame

    shutdown_camera_controller(controller, app)
    process_qt_events(app)


def test_camera_controller_error_and_finished_route_by_session():
    app = QApplication.instance() or QApplication([])
    controller = make_camera_controller(app)
    errors = []
    stopped = []
    controller.error.connect(errors.append)
    controller.stopped.connect(lambda: stopped.append(True))

    controller.start_preview()
    process_qt_events(app)
    thread = controller._thread
    session_id = controller.session_id()
    controller._on_worker_error(session_id - 1, "stale error")
    controller._on_worker_finished(session_id - 1)
    controller._on_worker_error(session_id, "current error")
    controller._on_worker_finished(session_id)

    assert errors == ["current error"]
    assert stopped == [True]
    assert controller.state() == CameraControllerState.FAILED.value
    assert controller.has_active_worker() is False
    if thread is not None:
        thread.quit()
        assert thread.wait(1000) is True


def test_normalize_reference_output_path_appends_png_extension():
    assert normalize_reference_output_path("data/reference/empty_reference") == Path("data/reference/empty_reference.png")


def test_normalize_reference_output_path_rejects_unsupported_extension():
    try:
        normalize_reference_output_path("data/reference/empty_reference.txt")
    except ValueError as exc:
        assert "Unsupported image extension" in str(exc)
    else:
        raise AssertionError("Expected unsupported extension to be rejected.")


def test_save_captured_reference_frame_creates_parent_and_saves_original_resolution(tmp_path):
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    output_path = tmp_path / "nested" / "reference.png"

    saved_path, resolution = save_captured_reference_frame(frame, output_path)

    assert saved_path == output_path
    assert resolution == (20, 12)
    loaded = cv2.imread(str(output_path))
    assert loaded is not None
    assert loaded.shape[:2] == (12, 20)


def test_save_captured_reference_frame_rejects_missing_frame(tmp_path):
    try:
        save_captured_reference_frame(None, tmp_path / "reference.png")
    except ValueError as exc:
        assert "No captured frame" in str(exc)
    else:
        raise AssertionError("Expected missing captured frame to be rejected.")


def test_aspect_image_label_stores_source_pixmap_and_resizes_safely():
    app = QApplication.instance() or QApplication([])
    widget = AspectImageLabel()
    pixmap = QPixmap(160, 90)
    pixmap.fill()

    widget.set_frame_pixmap(pixmap)
    widget.resize(320, 240)
    app.processEvents()

    assert widget._source_pixmap is not None
    assert widget._source_pixmap.size() == pixmap.size()


def test_aspect_image_label_clear_preview_removes_source_pixmap():
    app = QApplication.instance() or QApplication([])
    widget = AspectImageLabel()
    pixmap = QPixmap(16, 9)

    widget.set_frame_pixmap(pixmap)
    widget.clear_preview("Idle")

    assert widget._source_pixmap is None
    assert widget._message == "Idle"


def test_aspect_image_label_default_preview_remains_center_aligned():
    app = QApplication.instance() or QApplication([])
    widget = AspectImageLabel()
    pixmap = QPixmap(160, 90)
    pixmap.fill(QColor(0, 255, 0))

    widget.resize(240, 480)
    widget.show()
    widget.set_frame_pixmap(pixmap)
    app.processEvents()

    image = widget.grab().toImage()
    top_color = QColor(image.pixel(image.width() // 2, 20))
    center_color = QColor(image.pixel(image.width() // 2, image.height() // 2))

    assert not (top_color.green() > 200 and top_color.red() < 40 and top_color.blue() < 40)
    assert center_color.green() > 200


def test_aspect_image_label_repaints_non_black_frame_after_resize():
    app = QApplication.instance() or QApplication([])
    widget = AspectImageLabel()
    pixmap = QPixmap(160, 90)
    pixmap.fill(QColor(0, 0, 255))

    widget.resize(1, 1)
    widget.show()
    widget.set_frame_pixmap(pixmap)
    app.processEvents()
    widget.resize(320, 180)
    app.processEvents()

    image = widget.grab().toImage()
    center_color = QColor(image.pixel(image.width() // 2, image.height() // 2))

    assert widget._source_pixmap is not None
    assert not widget._source_pixmap.isNull()
    assert center_color.blue() > 200


def test_reference_capture_page_uses_compact_controls_and_workspace():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())

    panels = page.findChildren(SectionPanel)

    assert page.findChild(QSplitter, "referenceCaptureWorkspace") is not None
    assert page.findChild(QScrollArea, "referenceCaptureControlPaneScroll") is None
    assert page.findChild(QWidget, "referenceCaptureControlPane") is not None
    assert any(panel.property("density") == "compact" for panel in panels)
    assert isinstance(page.feedback_label, StatusBanner)
    assert isinstance(page.capture_state_banner, StatusBanner)
    assert isinstance(page.preview_label, AspectImageLabel)
    assert page.preview_label.minimumHeight() >= 480
    assert "Idle:" in page.capture_state_banner.text()
    assert page.capture_state_banner.property("state") == "idle"
    assert page.start_button.text() == "Start Camera"
    assert page.findChild(ActionButtonGrid, "actionButtonGrid") is not None
    assert page.capture_button.text() == "Capture Background"
    assert page.start_button.property("buttonRole") == "primary"
    assert page.output_path_edit.text() == str(job_reference_image_path("default_job"))
    assert page.output_path_edit.minimumWidth() == 0


def test_reference_capture_state_banner_tracks_capture_workflow():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())

    page._set_state("live_preview")
    assert "Starting camera:" in page.capture_state_banner.text()
    assert page.capture_state_banner.property("state") == "processing"
    assert page.capture_button.property("buttonRole") == "primary"

    page._set_state("captured_preview")
    assert "Captured:" in page.capture_state_banner.text()
    assert page.capture_state_banner.property("state") == "captured"
    assert page.save_button.property("buttonRole") == "primary"

    page._set_state("saved")
    assert "Saved:" in page.capture_state_banner.text()
    assert page.capture_state_banner.property("state") == "saved"


def test_reference_capture_preview_workspace_is_stable_when_feedback_grows():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())
    workspace = page.findChild(QSplitter, "referenceCaptureWorkspace")
    before = workspace.sizeHint()

    page._set_feedback("Saved reference. " + ("Long operator feedback. " * 40), ok=True)
    app.processEvents()

    assert workspace.sizeHint() == before


def test_reference_capture_live_frame_renders_visible_non_black_preview():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    page.resize(1366, 768)
    page.show()
    page._set_state("live_preview")
    page._camera_lifecycle = "starting"
    page._on_frame_ready(frame)
    app.processEvents()
    page.resize(1200, 720)
    app.processEvents()

    image = page.preview_label.grab().toImage()
    center_color = QColor(image.pixel(image.width() // 2, image.height() // 2))

    assert page.preview_label.isVisible()
    assert page.preview_label._source_pixmap is not None
    assert not page.preview_label._source_pixmap.isNull()
    assert center_color.blue() > 200
    assert center_color.red() < 40
    assert center_color.green() < 40

    page.close()


def test_reference_capture_stale_session_frame_is_ignored():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())
    current_frame = np.zeros((12, 20, 3), dtype=np.uint8)
    current_frame[:, :] = (255, 0, 0)
    stale_frame = np.zeros((12, 20, 3), dtype=np.uint8)
    stale_frame[:, :] = (0, 255, 0)

    page._camera_session_id = 2
    page._camera_lifecycle = "running"
    page._state = "live_preview"
    page._on_frame_ready_for_session(2, current_frame)
    source_before = page.preview_label._source_pixmap
    latest_before = page._latest_frame

    page._on_frame_ready_for_session(1, stale_frame)

    assert page._latest_frame is latest_before
    assert page.preview_label._source_pixmap is source_before


def test_reference_capture_blank_startup_frame_is_ignored():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    page._camera_session_id = 3
    page._camera_lifecycle = "starting"
    page._state = "live_preview"
    page._on_frame_ready_for_session(3, blank_frame)

    assert page._camera_lifecycle == "starting"
    assert page._latest_frame is None
    assert page.preview_label._source_pixmap is None
    assert page.capture_button.isEnabled() is False
    assert "Warming up camera:" in page.capture_state_banner.text()


def test_reference_capture_valid_frame_after_blank_startup_enables_capture():
    app = QApplication.instance() or QApplication([])
    page = ReferenceCapturePage(AppState())
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    valid_frame = np.full((720, 1280, 3), 80, dtype=np.uint8)

    page._camera_session_id = 4
    page._camera_lifecycle = "starting"
    page._state = "live_preview"
    page._on_frame_ready_for_session(4, blank_frame)
    page._on_frame_ready_for_session(4, valid_frame)

    assert page._camera_lifecycle == "running"
    assert page._latest_frame is valid_frame
    assert page.preview_label._source_pixmap is not None
    assert page.capture_button.isEnabled() is True


def test_default_zones_output_path_uses_app_state():
    state = AppState(zones_path=Path("custom/zones.json"))

    assert default_zones_output_path(state) == Path("custom/zones.json")


def test_default_zones_output_path_falls_back_to_job_zones():
    assert default_zones_output_path(AppState(inspection_job_name="Transistor")) == job_zones_path("transistor")


def test_normalize_zones_output_path_appends_json_extension():
    assert normalize_zones_output_path("zones_2") == Path("zones_2.json")


def test_normalize_zones_output_path_rejects_non_json_extension():
    try:
        normalize_zones_output_path("zones_2.txt")
    except ValueError as exc:
        assert "Unsupported zones file extension" in str(exc)
    else:
        raise AssertionError("Expected unsupported zones extension to be rejected.")


def test_compute_zone_canvas_style_uses_source_resolution():
    expected = {
        (640, 480): (12, 2, 7, 1),
        (1280, 720): (16, 2, 8, 1),
        (1024, 1024): (23, 3, 11, 2),
        (1920, 1080): (24, 3, 12, 2),
        (2560, 1440): (32, 4, 16, 3),
    }

    for (width, height), (label_font, polygon_width, point_size, point_pen) in expected.items():
        style = compute_zone_canvas_style(width, height)
        assert style.label_font_px == label_font
        assert style.finalized_pen_width == polygon_width
        assert style.current_pen_width == polygon_width
        assert style.point_diameter == point_size
        assert style.point_pen_width == point_pen


def test_compute_zone_canvas_style_is_deterministic_and_bounded():
    first = compute_zone_canvas_style(2560, 1440)
    second = compute_zone_canvas_style(2560, 1440)
    fallback = compute_zone_canvas_style(0, 0)

    assert first == second
    assert 2 <= first.finalized_pen_width <= 5
    assert 2 <= first.current_pen_width <= 5
    assert 7 <= first.point_diameter <= 18
    assert 1 <= first.point_pen_width <= 3
    assert 12 <= first.label_font_px <= 32
    assert fallback.finalized_pen_width == 2
    assert fallback.point_diameter == 7
    assert fallback.label_font_px == 12


def test_zone_editor_page_uses_compact_source_panel_and_canvas():
    app = QApplication.instance() or QApplication([])
    page = ZoneEditorPage(AppState())

    panels = page.findChildren(SectionPanel)

    assert page.findChild(QSplitter, "zoneEditorWorkspace") is not None
    assert page.findChild(QScrollArea, "zoneEditorControlPaneScroll") is None
    assert page.findChild(QWidget, "zoneEditorControlPane") is not None
    assert any(panel.property("density") == "compact" for panel in panels)
    assert isinstance(page.feedback_label, StatusBanner)
    assert isinstance(page.zone_state_banner, StatusBanner)
    assert page.view.minimumHeight() >= 360
    assert "Reference not loaded:" in page.zone_state_banner.text()
    assert page.zone_state_banner.property("state") == "blocked"
    buttons = {button.text() for button in page.findChildren(QPushButton)}
    assert {"Load Reference Image", "Fit Image", "Save Zones"}.issubset(buttons)
    assert page.findChild(ActionButtonGrid, "actionButtonGrid") is not None
    assert page._zone_canvas_style == compute_zone_canvas_style(0, 0)
    assert page.zones_path_edit.text() == str(job_zones_path("default_job"))


def test_zone_editor_load_reference_updates_canvas_style_and_labels(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image_path = tmp_path / "reference.png"
    cv2.imwrite(str(image_path), image)
    page = ZoneEditorPage(AppState())
    page.reference_path_edit.setText(str(image_path))

    page.load_reference_image()
    page._finalized_polygons.append([(100, 100), (500, 100), (500, 500)])
    page._current_points = [(120, 700), (700, 700)]
    page._redraw_overlays()

    expected_style = compute_zone_canvas_style(1024, 1024)
    labels = [item for item in page._overlay_items if item.__class__.__name__ == "QGraphicsTextItem"]
    ellipses = [item for item in page._overlay_items if item.__class__.__name__ == "QGraphicsEllipseItem"]
    polygons = [item for item in page._overlay_items if item.__class__.__name__ == "QGraphicsPolygonItem"]
    lines = [item for item in page._overlay_items if item.__class__.__name__ == "QGraphicsLineItem"]

    assert page._zone_canvas_style == expected_style
    assert labels
    assert labels[0].font().pixelSize() == expected_style.label_font_px
    assert polygons[0].pen().width() == expected_style.finalized_pen_width
    assert lines[0].pen().width() == expected_style.current_pen_width
    assert ellipses[0].rect().width() == expected_style.point_diameter
    assert ellipses[0].pen().width() == expected_style.point_pen_width


def test_zone_editor_state_banner_tracks_drawing_progress(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image_path = tmp_path / "reference.png"
    cv2.imwrite(str(image_path), image)
    page = ZoneEditorPage(AppState())
    page.reference_path_edit.setText(str(image_path))

    page.load_reference_image()
    assert "Reference loaded:" in page.zone_state_banner.text()
    assert page.zone_state_banner.property("state") == "ready"

    page._add_point(10, 10)
    assert "Polygon in progress:" in page.zone_state_banner.text()
    assert page.zone_state_banner.property("state") == "live"

    page._add_point(30, 10)
    page._add_point(30, 30)
    page.finish_polygon()
    assert "Zones ready:" in page.zone_state_banner.text()
    assert page.zone_state_banner.property("state") == "ready"


def test_prepare_polygons_for_save_auto_finalizes_valid_current_polygon():
    polygons, warning = prepare_polygons_for_save([], [(1, 1), (10, 1), (10, 10)])

    assert polygons == [[(1, 1), (10, 1), (10, 10)]]
    assert warning is None


def test_prepare_polygons_for_save_skips_invalid_current_polygon_when_finalized_exists():
    polygons, warning = prepare_polygons_for_save([[(1, 1), (10, 1), (10, 10)]], [(5, 5), (6, 6)])

    assert polygons == [[(1, 1), (10, 1), (10, 10)]]
    assert warning == "Unfinished polygon has fewer than 3 points and was not saved."


def test_prepare_polygons_for_save_rejects_empty_session():
    try:
        prepare_polygons_for_save([], [])
    except ValueError as exc:
        assert "No valid polygon to save" in str(exc)
    else:
        raise AssertionError("Expected empty zone session to be rejected.")


def test_save_zone_session_writes_backend_compatible_zone_json(tmp_path):
    zones_path = tmp_path / "nested" / "zones.json"

    saved_path, count, warning = save_zone_session(
        zones_path,
        image_width=100,
        image_height=50,
        finalized_polygons=[[(1, 2), (20, 2), (20, 25)]],
        current_points=[],
    )

    assert saved_path == zones_path
    assert count == 1
    assert warning is None
    zone_config = load_zones(zones_path)
    assert zone_config.image_width == 100
    assert zone_config.image_height == 50
    assert zone_config.zones[0].points == [(1, 2), (20, 2), (20, 25)]


def test_save_zone_session_creates_parent_and_auto_finalizes_current_polygon(tmp_path):
    zones_path = tmp_path / "nested" / "zones.json"

    saved_path, count, warning = save_zone_session(
        zones_path,
        image_width=30,
        image_height=30,
        finalized_polygons=[],
        current_points=[(1, 1), (20, 1), (20, 20)],
    )

    assert saved_path == zones_path
    assert count == 1
    assert warning is None
    assert zones_path.exists()


def test_save_zone_session_normalizes_typed_json_filename(tmp_path):
    zones_path = tmp_path / "zones_2"

    saved_path, count, warning = save_zone_session(
        zones_path,
        image_width=30,
        image_height=30,
        finalized_polygons=[],
        current_points=[(1, 1), (20, 1), (20, 20)],
    )

    assert saved_path == tmp_path / "zones_2.json"
    assert count == 1
    assert warning is None
    assert saved_path.exists()


def test_zone_editor_custom_zones_path_updates_state_and_saves_requested_file(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image_path = tmp_path / "reference.png"
    cv2.imwrite(str(image_path), image)
    state = AppState()
    saved_paths = []
    page = ZoneEditorPage(state, zones_saved_callback=lambda path: saved_paths.append(path))
    custom_path = tmp_path / "zones_2"

    page.reference_path_edit.setText(str(image_path))
    page.zones_path_edit.setText(str(custom_path))
    assert state.zones_path == custom_path
    assert state.zones_path_auto is False

    page.load_reference_image()
    page._current_points = [(10, 10), (40, 10), (40, 40)]
    page.save_zones()

    expected_path = tmp_path / "zones_2.json"
    assert expected_path.exists()
    assert page.zones_path_edit.text() == str(expected_path)
    assert state.zones_path == expected_path
    assert saved_paths == [expected_path]
    assert "Saved 1 polygon" in page.feedback_label.text()


def test_main_window_callbacks_keep_setup_pages_synchronized(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppState())
    reference_path = tmp_path / "reference.png"
    zones_path = tmp_path / "zones.json"

    window._on_reference_saved(reference_path)
    window._on_zones_saved(zones_path)

    assert window._project_setup_page.reference_path_edit.text() == str(reference_path)
    assert window._project_setup_page.zones_path_edit.text() == str(zones_path)
    assert window._zone_editor_page.reference_path_edit.text() == str(reference_path)


def test_zone_editor_canvas_workspace_is_stable_when_feedback_grows():
    app = QApplication.instance() or QApplication([])
    page = ZoneEditorPage(AppState())
    workspace = page.findChild(QSplitter, "zoneEditorWorkspace")
    before = workspace.sizeHint()

    page._set_feedback("Zone save status. " + ("Long path or validation message. " * 40), ok=True)
    app.processEvents()

    assert workspace.sizeHint() == before


def test_app_locale_formats_spinbox_with_arabic_numerals():
    app = QApplication.instance() or QApplication([])
    configure_app_locale(app)
    spinbox = QSpinBox()
    spinbox.setLocale(APP_LOCALE)
    spinbox.setRange(0, 1000)
    spinbox.setValue(500)

    assert spinbox.text() == "500"
    assert not any("\u0e50" <= char <= "\u0e59" for char in spinbox.text())


def test_default_inspect_output_dir_uses_app_state():
    assert default_inspect_output_dir(AppState(image_output_dir=Path("outputs/custom"))) == Path("outputs/custom")


def test_default_inspect_output_dir_falls_back_to_job_image_output():
    assert default_inspect_output_dir(AppState(inspection_job_name="Transistor")) == Path("outputs/transistor/image")


def test_default_camera_output_dir_uses_app_state():
    assert default_camera_output_dir(AppState(camera_output_dir=Path("outputs/camera"))) == Path("outputs/camera")


def test_default_camera_output_dir_falls_back_to_job_camera_output():
    assert default_camera_output_dir(AppState(inspection_job_name="Transistor")) == Path("outputs/transistor/camera")


def test_captured_image_path_uses_readable_timestamp_under_captured_folder():
    from datetime import datetime

    path = captured_image_path("outputs/camera", datetime(2026, 5, 14, 9, 8, 7, 123456))

    assert path == Path("outputs/camera/captured/capture_20260514_090807_123456.png")


def test_save_captured_inspection_frame_creates_parent_and_saves_original_resolution(tmp_path):
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)

    saved_path, resolution = save_captured_inspection_frame(frame, tmp_path / "camera_run")

    assert saved_path.parent == tmp_path / "camera_run" / "captured"
    assert saved_path.suffix == ".png"
    assert resolution == (20, 12)
    loaded = cv2.imread(str(saved_path))
    assert loaded is not None
    assert loaded.shape[:2] == (12, 20)


def test_save_captured_inspection_frame_rejects_missing_frame(tmp_path):
    try:
        save_captured_inspection_frame(None, tmp_path)
    except ValueError as exc:
        assert "No captured frame" in str(exc)
    else:
        raise AssertionError("Expected missing captured frame to be rejected.")


def test_require_saved_config_path_rejects_missing_config():
    try:
        require_saved_config_path(AppState())
    except ValueError as exc:
        assert "Open or save a config" in str(exc)
    else:
        raise AssertionError("Expected missing config path to be rejected.")


def test_require_saved_config_path_accepts_existing_config(tmp_path):
    path = tmp_path / "local_inspection.yaml"
    path.write_text("model: {}\n", encoding="utf-8")

    assert require_saved_config_path(AppState(config_path=path)) == path


def test_gui_inspection_config_disables_show_images_without_mutating_loaded_config(tmp_path):
    data = default_config_data()
    data["model"]["path"] = "model.pt"
    data["output"]["show_images"] = True
    config_path = tmp_path / "local_inspection.yaml"
    save_config_data(config_path, data)
    config = load_config(config_path)

    gui_config = config_for_gui_inspection(config)

    assert config.output.show_images is True
    assert gui_config.output.show_images is False


def test_inspect_image_page_summary_widgets_exist():
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    dimensions = theme_dimensions()

    assert page.inputs_panel.property("density") == "compact"
    assert page.inputs_panel.property("styleVariant") == "classic"
    assert page.inputs_panel.findChild(QLabel).text() == "Input Image"
    assert isinstance(page.workspace, WorkbenchLayout)
    assert page.findChild(QScrollArea, "inspectImageLeftPaneScroll") is None
    assert page.findChild(QWidget, "inspectImageLeftPane") is not None
    assert page.findChild(QWidget, "inspectImageSecondaryPane") is not None
    assert isinstance(page.summary_panel, ResultSummaryPanel)
    assert page.summary_panel.csv_label.text() == "CSV Log Path"
    assert page.summary_panel.error_label.text() == "Error Message"
    assert isinstance(page.result_image_tabs, ResultImageTabs)
    assert isinstance(page.feedback_label, StatusBanner)
    assert isinstance(page.operation_state_banner, StatusBanner)
    assert "Blocked:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "blocked"
    assert page.run_button.text() == "Run Inspection"
    assert page.open_output_button.text() == "Open Output Folder"
    action_section = page.findChild(ActionSection, "actionSection")
    assert action_section is not None
    assert page.run_button.minimumHeight() >= dimensions.button_min_height
    assert action_section.minimumHeight() >= page.run_button.sizeHint().height()
    assert page.summary_panel.minimumHeight() == dimensions.inspect_summary_min_height
    assert page.result_image_tabs.minimumHeight() == dimensions.inspect_result_image_tabs_min_height
    assert page.tabs.minimumHeight() == dimensions.inspect_result_tabs_min_height
    assert page.annotated_preview.minimumHeight() == dimensions.inspect_result_preview_min_height
    assert has_ancestor(page.run_button, QScrollArea) is False
    assert has_ancestor(page.operation_state_banner, QScrollArea) is False
    assert has_ancestor(page.summary_panel, QScrollArea) is False
    assert has_ancestor(page.config_path_edit, QScrollArea) is False
    assert page.findChild(QWidget, "inspectImageLeftPane").maximumWidth() == dimensions.workflow_rail_size_hint_width
    assert page.workspace.widget(1).sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_inspect_image_run_button_is_allocated_full_height_at_desktop_size():
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    page.resize(1366, 768)
    page.show()
    app.processEvents()

    try:
        assert page.run_button.height() >= page.run_button.minimumHeight()
        assert page.run_button.height() >= theme_dimensions().button_min_height
        assert has_ancestor(page.run_button, QScrollArea) is False
        assert has_ancestor(page.operation_state_banner, QScrollArea) is False
    finally:
        page.close()


def test_inspect_image_page_operation_state_transitions(tmp_path):
    app = QApplication.instance() or QApplication([])
    config_path = tmp_path / "local_inspection.yaml"
    image_path = tmp_path / "part.png"
    output_dir = tmp_path / "out"
    config_path.write_text("model: {}\n", encoding="utf-8")
    cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    page = InspectImagePage(AppState(config_path=config_path))

    assert "Waiting:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "waiting"

    page.image_path_edit.setText(str(image_path))
    page.output_dir_edit.setText(str(output_dir))
    page._refresh_operation_state()

    assert "Ready:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "ready"

    page._set_running(True)
    assert "Inspecting:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "processing"
    assert page.operation_state_banner._spinner_timer.isActive() is True

    page._on_completed(InspectionResult(image_path=str(image_path), final_result=FinalResult.OK), output_dir)
    assert "Complete:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "complete"

    page._thread = None
    page._set_running(False)


def test_inspect_image_result_completion_keeps_stable_workspace(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    result = InspectionResult(image_path="image.png", final_result=FinalResult.OK)
    workspace_index = page.layout().indexOf(page.workspace)

    page._on_completed(result, tmp_path)

    assert page.layout().indexOf(page.workspace) == workspace_index
    assert page.workspace.widget(0) is page.findChild(QWidget, "inspectImageLeftPane")
    assert page.findChild(QWidget, "inspectImageSecondaryPane").parentWidget() is page.findChild(QWidget, "inspectImageLeftPane")
    assert page.workspace.widget(1).findChild(ResultImageTabs) is page.result_image_tabs
    assert page.summary_panel.result_badge.text() == "OK"


def test_image_preview_widget_accepts_and_clears_image(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = np.zeros((10, 16, 3), dtype=np.uint8)
    image_path = tmp_path / "preview.png"
    cv2.imwrite(str(image_path), image)
    widget = ImagePreviewWidget("No image")

    widget.set_image_path(image_path, "Missing")
    assert widget._pixmap_item is not None
    assert widget._zoom == 1.0

    widget.clear_preview("Cleared")
    assert widget._pixmap_item is None
    assert widget._text_item is not None


def test_image_preview_widget_size_hint_stays_bounded_after_large_image_load(tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "large_preview.png"
    cv2.imwrite(str(image_path), np.zeros((1600, 2400, 3), dtype=np.uint8))
    widget = ImagePreviewWidget("No image", min_height=340)
    before_size_hint = widget.sizeHint()
    before_minimum_hint = widget.minimumSizeHint()

    widget.set_image_path(image_path, "Missing")
    app.processEvents()

    assert widget._pixmap_item is not None
    assert widget.sizeHint() == before_size_hint
    assert widget.minimumSizeHint() == before_minimum_hint


def test_image_preview_widget_fit_method_is_safe_without_image():
    app = QApplication.instance() or QApplication([])
    widget = ImagePreviewWidget("No image")

    widget.fit_current_image()

    assert widget._pixmap_item is None


def test_inspect_image_page_hides_error_message_when_empty(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    result = InspectionResult(
        image_path="image.png",
        final_result=FinalResult.NG,
        presence_status="PART_PRESENT",
        foreground_ratio=0.25,
        mean_diff=12.0,
        largest_blob_area=550.0,
        anomaly_pred_label=True,
        anomaly_score=0.9,
        total_time_ms=43234.3253463,
        presence_time_ms=5.0,
        anomaly_time_ms=10.0,
    )

    page._update_summary(result, tmp_path)

    assert page.summary_panel.error_text.isHidden() is True
    assert page.summary_panel.error_label.isHidden() is True
    assert page.summary_panel.essential_metric_grid.value_text("Presence") == "Part present"
    assert page.summary_panel.essential_metric_grid.value_text("Anomaly decision") == "Anomaly detected"
    assert page.summary_panel.essential_metric_grid.value_text("Score") == "0.9"
    assert page.summary_panel.essential_metric_grid.value_text("Inspection time") == "43.2 s"
    assert page.summary_panel.technical_metric_grid.value_text("Foreground") == "25.0%"
    assert page.summary_panel.technical_metric_grid.value_text("Blob area") == "550 px^2"
    assert page.summary_panel.technical_metric_grid.value_text("Mean pixel difference") == "12.0"
    assert page.summary_panel.technical_metric_grid.value_text("Presence check time") == "5 ms"


def test_inspect_image_page_shows_error_message_when_present(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    result = InspectionResult(image_path="image.png", final_result=FinalResult.ERROR, error_message="failure")

    page._update_summary(result, tmp_path)

    assert page.summary_panel.error_text.isHidden() is False
    assert page.summary_panel.error_label.isHidden() is False
    assert page.summary_panel.error_text.toPlainText() == "failure"


def test_fit_image_button_exists():
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())

    assert page.fit_image_button.text() == "Fit Image"


def test_inspect_image_result_images_section_has_useful_natural_height():
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    dimensions = theme_dimensions()

    assert page.tabs.minimumHeight() == dimensions.inspect_result_tabs_min_height
    assert page.annotated_preview.minimumHeight() == dimensions.inspect_result_preview_min_height
    assert page.tabs.minimumHeight() < dimensions.result_tabs_min_height
    assert page.annotated_preview.minimumHeight() < dimensions.preview_min_height
    assert page.workspace.minimumSizeHint().height() <= 560


def test_inspect_image_workspace_size_hint_stays_stable_after_artifact_images_load(tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "large_result.png"
    cv2.imwrite(str(image_path), np.zeros((1600, 2400, 3), dtype=np.uint8))
    page = InspectImagePage(AppState())
    before = page.workspace.sizeHint()

    page.result_image_tabs.set_artifacts(
        annotated_path=image_path,
        heatmap_path=image_path,
        presence_mask_path=image_path,
    )
    app.processEvents()

    assert page.workspace.sizeHint() == before
    assert page.annotated_preview.sizeHint().height() == theme_dimensions().inspect_result_preview_min_height


def test_inspect_image_workspace_is_stable_when_technical_details_expand(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())
    before = page.workspace.sizeHint()
    metrics = (
        ("Presence", "Part present"),
        ("Score", "0.1"),
        ("Inspection time", "15 ms"),
        ("Backend", "backend"),
        ("Mean pixel difference", "12.0"),
    )

    page.summary_panel.set_result(
        "OK",
        metrics,
        csv_log_path=str(tmp_path / "inspection_log.csv"),
        error_message="",
    )
    page.summary_panel.technical_button.setChecked(True)
    app.processEvents()

    assert page.workspace.sizeHint() == before


def test_inspect_image_default_output_tracks_job_until_manual_override():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = InspectImagePage(state)

    state.set_inspection_job_name("Transistor")
    page.refresh_from_state()

    assert normalized_path_text(page.output_dir_edit.text()) == "outputs/transistor/image"

    state.image_output_dir = Path("outputs/manual_image")
    state.set_inspection_job_name("Metal Surface")
    page.refresh_from_state()

    assert normalized_path_text(page.output_dir_edit.text()) == "outputs/manual_image"


def test_inspect_image_text_edit_marks_manual_output_override():
    app = QApplication.instance() or QApplication([])
    state = AppState(inspection_job_name="Transistor")
    page = InspectImagePage(state)

    page.output_dir_edit.setText("outputs/custom_image")
    page._mark_output_dir_override("outputs/custom_image")
    state.set_inspection_job_name("Metal Surface")
    page.refresh_from_state()

    assert state.image_output_dir == Path("outputs/custom_image")
    assert normalized_path_text(page.output_dir_edit.text()) == "outputs/custom_image"


def test_inspect_camera_page_constructs_with_expected_controls():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())

    assert page.start_button.text() == "Start Camera"
    assert page.capture_button.text() == "Capture Part"
    assert page.inspect_button.text() == "Inspect Captured Image"
    assert page.tabs.count() == 3


def test_inspect_camera_page_uses_shared_ui_components():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    dimensions = theme_dimensions()

    panels = page.findChildren(SectionPanel)
    section_titles = {label.text() for label in page.findChildren(QLabel) if label.objectName() == "sectionPanelTitle"}

    assert len(panels) >= 4
    assert {"Camera", "Operation", "Result Summary", "Config and Output"}.issubset(section_titles)
    assert any(panel.property("density") == "compact" for panel in panels)
    assert isinstance(page.findChild(QSplitter, "cameraOperatorWorkspace"), WorkbenchLayout)
    assert page.findChild(QScrollArea, "cameraControlPaneScroll") is None
    assert page.findChild(QWidget, "cameraControlPane") is not None
    assert page.findChild(QWidget, "cameraSecondaryPane") is not None
    assert isinstance(page.visual_stack, QStackedWidget)
    assert isinstance(page.feedback_label, StatusBanner)
    assert isinstance(page.operation_state_banner, StatusBanner)
    assert isinstance(page.summary_panel, ResultSummaryPanel)
    assert isinstance(page.result_image_tabs, ResultImageTabs)
    assert page.summary_panel.error_label.text() == "Error Message"
    assert page.fit_image_button.text() == "Fit Image"
    assert "Blocked:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "blocked"
    assert page.findChild(QWidget, "cameraPreviewContainer") is not None
    assert page.preview_container.minimumHeight() == dimensions.camera_visual_preview_min_height
    assert page.result_image_tabs.minimumHeight() == dimensions.camera_result_image_tabs_min_height
    assert page.tabs.minimumHeight() == dimensions.camera_result_tabs_min_height
    assert page.annotated_preview.minimumHeight() == dimensions.camera_result_preview_min_height
    assert page.findChild(QWidget, "cameraControlPane").maximumWidth() == dimensions.workflow_rail_size_hint_width
    assert page.findChild(ActionButtonGrid, "actionButtonGrid") is not None
    assert has_ancestor(page.start_button, QScrollArea) is False
    assert has_ancestor(page.stop_button, QScrollArea) is False
    assert has_ancestor(page.capture_button, QScrollArea) is False
    assert has_ancestor(page.inspect_button, QScrollArea) is False
    assert has_ancestor(page.operation_state_banner, QScrollArea) is False
    assert has_ancestor(page.summary_panel, QScrollArea) is False
    assert has_ancestor(page.config_path_edit, QScrollArea) is False
    assert page.findChild(QSplitter, "cameraOperatorWorkspace").widget(1).sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_inspect_camera_action_buttons_are_allocated_full_height_at_desktop_size():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    page.resize(1366, 768)
    page.show()
    app.processEvents()

    try:
        for button in (
            page.start_button,
            page.stop_button,
            page.capture_button,
            page.inspect_button,
            page.retake_button,
            page.open_output_button,
        ):
            assert button.height() >= button.minimumHeight()
            assert button.height() >= theme_dimensions().button_min_height
            assert has_ancestor(button, QScrollArea) is False
        assert has_ancestor(page.operation_state_banner, QScrollArea) is False
    finally:
        page.close()


def test_inspect_camera_visual_workspace_switches_between_preview_and_results():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())

    assert page.visual_stack.currentWidget() is page.preview_container

    page._set_state("captured_preview")
    assert page.visual_stack.currentWidget() is page.preview_container

    page._set_state("result_ready")
    assert page.visual_stack.currentWidget() is page.result_image_tabs

    page._set_state("live_preview")
    assert page.visual_stack.currentWidget() is page.preview_container


def test_inspect_camera_result_workspace_size_hint_stays_stable_after_artifact_images_load(tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "large_camera_result.png"
    cv2.imwrite(str(image_path), np.zeros((1600, 2400, 3), dtype=np.uint8))
    page = InspectCameraPage(AppState())
    before = page.findChild(QSplitter, "cameraOperatorWorkspace").sizeHint()

    page.result_image_tabs.set_artifacts(
        annotated_path=image_path,
        heatmap_path=image_path,
        presence_mask_path=image_path,
    )
    app.processEvents()

    assert page.findChild(QSplitter, "cameraOperatorWorkspace").sizeHint() == before
    assert page.annotated_preview.sizeHint().height() == theme_dimensions().camera_result_preview_min_height


def test_inspect_camera_workspace_is_stable_when_result_details_expand(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    workspace = page.findChild(QSplitter, "cameraOperatorWorkspace")
    before = workspace.sizeHint()
    metrics = (
        ("Presence", "Part present"),
        ("Score", "0.9"),
        ("Inspection time", "18 ms"),
        ("Backend", "backend"),
        ("Mean pixel difference", "20.0"),
    )

    page.summary_panel.set_result(
        "NG",
        metrics,
        csv_log_path=str(tmp_path / "camera" / "inspection_log.csv"),
        error_message="Inspection detail message. " * 20,
    )
    page.summary_panel.technical_button.setChecked(True)
    app.processEvents()

    assert workspace.sizeHint() == before


def test_inspect_camera_operation_state_banner_and_primary_actions(tmp_path):
    app = QApplication.instance() or QApplication([])
    config_path = tmp_path / "local_inspection.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    page = InspectCameraPage(AppState(config_path=config_path))

    assert "Ready:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "ready"
    assert page.start_button.property("buttonRole") == "primary"

    page._set_state("live_preview")
    assert "Starting camera:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "processing"
    assert page.capture_button.property("buttonRole") == "primary"

    page._latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    page._update_operation_state_banner("live_preview")
    assert "Camera live:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "live"

    page._set_state("captured_preview")
    assert "Ready to inspect:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "captured"
    assert page.inspect_button.property("buttonRole") == "primary"

    page._set_state("inspecting")
    assert "Inspecting:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "processing"
    assert page.operation_state_banner._spinner_timer.isActive() is True
    assert page.open_output_button.isEnabled() is False

    page._set_state("result_ready")
    assert "Complete:" in page.operation_state_banner.text()
    assert page.operation_state_banner.property("state") == "complete"
    assert page.retake_button.property("buttonRole") == "primary"


def test_inspect_camera_live_frame_updates_visible_preview_widget():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 160)

    page.visual_stack.setCurrentWidget(page.result_image_tabs)
    page._state = "live_preview"
    page._on_controller_frame_ready(frame)

    assert page.visual_stack.currentWidget() is page.preview_container
    assert page.preview_label._source_pixmap is not None
    assert page.preview_label._source_pixmap.width() == 20
    assert page.preview_label._source_pixmap.height() == 12
    assert page._latest_frame is frame


def test_inspect_camera_start_button_calls_camera_controller(monkeypatch):
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    calls = []

    monkeypatch.setattr(page.camera_controller, "start_preview", lambda: calls.append("start"))

    page.start_camera()

    assert calls == ["start"]
    assert page._latest_frame is None
    assert page.visual_stack.currentWidget() is page.preview_container


def test_inspect_camera_stop_button_calls_camera_controller(monkeypatch):
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    calls = []
    page._state = "live_preview"
    page._latest_frame = np.full((12, 20, 3), 80, dtype=np.uint8)
    monkeypatch.setattr(page.camera_controller, "stop_preview", lambda: calls.append("stop"))

    page.stop_camera()

    assert calls == ["stop"]
    assert page._state == "idle"
    assert page._latest_frame is None
    assert page.preview_label._source_pixmap is None


def test_inspect_camera_capture_stays_disabled_until_controller_valid_frame():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    valid_frame = np.full((720, 1280, 3), 80, dtype=np.uint8)

    page._on_camera_controller_state_changed(CameraControllerState.WARMING.value, "Warming up camera...", "info")
    assert page.capture_button.isEnabled() is False

    page.camera_controller._set_state(CameraControllerState.RUNNING, "Camera live.", "success")
    page._on_controller_frame_ready(valid_frame)

    assert page._latest_frame is valid_frame
    assert page.preview_label._source_pixmap is not None
    assert page.capture_button.isEnabled() is True


def test_inspect_camera_controller_warning_updates_operation_feedback():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())

    page._on_camera_controller_state_changed(
        CameraControllerState.RECOVERING.value,
        "Blank startup frames detected; reopening camera...",
        "warning",
    )

    assert page._latest_frame is None
    assert page.capture_button.isEnabled() is False
    assert "reopening camera" in page.feedback_label.text()
    assert page.operation_state_banner.property("level") == "warning"


def test_inspect_camera_controller_error_returns_to_idle_and_enables_start():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    page._state = "live_preview"

    page._on_controller_error(
        "Camera opened but frames stayed blank. Please retry or check exposure, lens, privacy shutter, or driver.",
    )

    assert page._state == "idle"
    assert page.start_button.isEnabled() is True
    assert "frames stayed blank" in page.feedback_label.text()


def test_inspect_camera_controller_stopped_does_not_clear_non_preview_state():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    page._set_state("captured_preview")

    page._on_controller_stopped()

    assert page._state == "captured_preview"


def test_inspect_camera_full_page_live_frame_renders_visible_non_black_preview():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    page.resize(1366, 768)
    page.show()
    page._state = "live_preview"
    page._on_controller_frame_ready(frame)
    app.processEvents()
    page.resize(1200, 720)
    app.processEvents()

    image = page.preview_label.grab().toImage()
    center_color = QColor(image.pixel(image.width() // 2, image.height() // 2))

    assert page.visual_stack.currentWidget() is page.preview_container
    assert page.preview_label.isVisible()
    assert page.preview_label._source_pixmap is not None
    assert not page.preview_label._source_pixmap.isNull()
    assert center_color.green() > 200
    assert center_color.red() < 40
    assert center_color.blue() < 40

    page.close()


def test_inspect_camera_live_preview_paints_centered_frame_in_tall_workspace():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    page.visual_stack.resize(260, 640)
    page.visual_stack.show()
    page._state = "live_preview"
    page._on_controller_frame_ready(frame)
    app.processEvents()

    image = page.preview_label.grab().toImage()
    center_color = QColor(image.pixel(image.width() // 2, image.height() // 2))

    assert page.visual_stack.currentWidget() is page.preview_container
    assert page.preview_label.geometry().top() > 0
    assert page.preview_label.geometry().height() < page.preview_container.height()
    assert center_color.green() > 200
    assert center_color.red() < 40
    assert center_color.blue() < 40


def test_inspect_camera_capture_state_uses_preview_visual_workspace():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    frame[:, :] = (90, 40, 10)
    page._latest_frame = frame

    page.capture_part()

    assert page._state == "captured_preview"
    assert page.visual_stack.currentWidget() is page.preview_container
    assert page.preview_label._source_pixmap is not None


def test_inspect_camera_preserves_captured_frame_after_preview_stop(monkeypatch):
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState(config_path=Path("configs/local_inspection.yaml")))
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    frame[:, :] = (90, 40, 10)
    page._latest_frame = frame
    monkeypatch.setattr(page.camera_controller, "stop_preview", lambda: None)

    page.capture_part()
    captured = page._captured_frame.copy()
    page.stop_camera()

    assert page._captured_frame is not None
    assert np.array_equal(page._captured_frame, captured)
    assert page._state == "captured_preview"
    assert page.inspect_button.isEnabled() is True


def test_inspect_camera_inspects_preserved_capture_after_preview_stop(tmp_path, monkeypatch):
    import anomaly_inspection.desktop.pages.inspect_camera as camera_page_module

    app = QApplication.instance() or QApplication([])
    config_path = tmp_path / "local_inspection.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    output_dir = tmp_path / "camera_output"
    created_workers = []

    class ImmediateThread(QObject):
        started = Signal()
        finished = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)

        def start(self):
            self.started.emit()

        def quit(self):
            self.finished.emit()

        def wait(self, timeout=None):
            return True

    class FakeInspectionWorker(QObject):
        completed = Signal(object, object)
        failed = Signal(str)
        cancelled = Signal(str)
        status = Signal(str)
        finished = Signal()

        def __init__(
            self,
            config_path,
            image_path,
            output_dir,
            inspection_mode="image",
            runtime_manager=None,
        ):
            super().__init__()
            self.config_path = Path(config_path)
            self.image_path = Path(image_path)
            self.output_dir = Path(output_dir)
            self.inspection_mode = inspection_mode
            self.runtime_manager = runtime_manager
            created_workers.append(self)

        def moveToThread(self, thread):
            self.thread = thread

        def run(self):
            self.finished.emit()

        def request_cancel(self):
            self.cancelled.emit("Cancelled")

    monkeypatch.setattr(camera_page_module, "QThread", ImmediateThread)
    monkeypatch.setattr(camera_page_module, "InspectionWorker", FakeInspectionWorker)

    page = InspectCameraPage(AppState(config_path=config_path))
    page.output_dir_edit.setText(str(output_dir))
    frame = np.zeros((10, 14, 3), dtype=np.uint8)
    frame[:, :] = (11, 22, 33)
    page._latest_frame = frame
    monkeypatch.setattr(page.camera_controller, "stop_preview", lambda: None)

    page.capture_part()
    page.stop_camera()
    page.inspect_captured_image()

    assert len(created_workers) == 1
    worker = created_workers[0]
    assert worker.config_path == config_path
    assert worker.output_dir == output_dir
    assert worker.inspection_mode == "camera"
    assert worker.image_path.parent == output_dir / "captured"
    assert worker.image_path.exists()
    saved = cv2.imread(str(worker.image_path), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == frame.shape
    assert np.array_equal(saved, frame)


def test_inspect_camera_retake_after_error_result_returns_to_preview_visual_workspace():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())

    page._set_state("result_ready")
    assert page.visual_stack.currentWidget() is page.result_image_tabs

    page.camera_controller._set_state(CameraControllerState.RUNNING, "Camera live.", "success")
    page.retake()

    assert page._state == "live_preview"
    assert page.visual_stack.currentWidget() is page.preview_container
    assert "Retaking" in page.feedback_label.text()


def test_inspect_camera_default_output_tracks_job_until_manual_override():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = InspectCameraPage(state)

    state.set_inspection_job_name("Transistor")
    page.refresh_from_state()

    assert normalized_path_text(page.output_dir_edit.text()) == "outputs/transistor/camera"

    state.camera_output_dir = Path("outputs/manual_camera")
    state.set_inspection_job_name("Metal Surface")
    page.refresh_from_state()

    assert normalized_path_text(page.output_dir_edit.text()) == "outputs/manual_camera"


def test_inspect_camera_text_edit_marks_manual_output_override():
    app = QApplication.instance() or QApplication([])
    state = AppState(inspection_job_name="Transistor")
    page = InspectCameraPage(state)

    page.output_dir_edit.setText("outputs/custom_camera")
    page._mark_output_dir_override("outputs/custom_camera")
    state.set_inspection_job_name("Metal Surface")
    page.refresh_from_state()

    assert state.camera_output_dir == Path("outputs/custom_camera")
    assert normalized_path_text(page.output_dir_edit.text()) == "outputs/custom_camera"


def test_mode_specific_output_overrides_do_not_cross_pages():
    state = AppState(inspection_job_name="Transistor", camera_output_dir=Path("outputs/manual_camera"))

    assert default_camera_output_dir(state) == Path("outputs/manual_camera")
    assert default_inspect_output_dir(state) == Path("outputs/transistor/image")

    state = AppState(inspection_job_name="Transistor", image_output_dir=Path("outputs/manual_image"))

    assert default_inspect_output_dir(state) == Path("outputs/manual_image")
    assert default_camera_output_dir(state) == Path("outputs/transistor/camera")


def test_inspect_camera_page_refuses_inspection_without_captured_frame(tmp_path):
    app = QApplication.instance() or QApplication([])
    config_path = tmp_path / "local_inspection.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    page = InspectCameraPage(AppState(config_path=config_path, output_dir=tmp_path / "out"))
    page.output_dir_edit.setText(str(tmp_path / "out"))

    page.inspect_captured_image()

    assert "No captured frame" in page.feedback_label.text()


def test_inspect_camera_page_disables_inspection_without_saved_config():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())

    page._set_state("captured_preview")

    assert page.inspect_button.isEnabled() is False


def test_inspect_camera_shutdown_is_safe_when_no_workers_exist():
    app = QApplication.instance() or QApplication([])
    page = InspectCameraPage(AppState())

    page.shutdown()
    page.shutdown()

    assert page.camera_controller.has_active_worker() is False
    assert page._inspection_thread is None


def test_inspect_image_shutdown_is_safe_when_no_thread_exists():
    app = QApplication.instance() or QApplication([])
    page = InspectImagePage(AppState())

    page.shutdown()
    page.shutdown()

    assert page._thread is None
    assert page._worker is None


def test_inspection_page_worker_slots_have_gui_thread_affinity():
    app = QApplication.instance() or QApplication([])
    image_page = InspectImagePage(AppState())
    camera_page = InspectCameraPage(AppState())
    observed_threads = []

    class StatusEmitter(QObject):
        status = Signal(str)
        finished = Signal()

        @Slot()
        def run(self):
            self.status.emit("Worker status")
            self.finished.emit()

    emitter = StatusEmitter()
    thread = QThread()
    emitter.moveToThread(thread)
    original_image_feedback = image_page._set_feedback
    original_camera_feedback = camera_page._set_feedback

    def record_image_feedback(message, ok):
        observed_threads.append(QThread.currentThread())
        original_image_feedback(message, ok)

    def record_camera_feedback(message, ok):
        observed_threads.append(QThread.currentThread())
        original_camera_feedback(message, ok)

    image_page._set_feedback = record_image_feedback
    camera_page._set_feedback = record_camera_feedback
    emitter.status.connect(image_page._on_worker_status, Qt.ConnectionType.QueuedConnection)
    emitter.status.connect(camera_page._on_inspection_status, Qt.ConnectionType.QueuedConnection)
    emitter.finished.connect(emitter.deleteLater)
    emitter.finished.connect(thread.quit)
    thread.started.connect(emitter.run)
    thread.start()
    for _ in range(1000):
        app.processEvents()
        if thread.isFinished() and len(observed_threads) == 2:
            break
        QThread.msleep(1)

    assert observed_threads == [app.thread(), app.thread()]
    assert thread.wait(1000) is True


def write_synthetic_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp",
        "image_name",
        "image_path",
        "final_result",
        "presence_status",
        "foreground_ratio",
        "mean_diff",
        "largest_blob_area",
        "changed_pixel_count",
        "zone_pixel_count",
        "anomaly_ran",
        "anomaly_backend",
        "anomaly_pred_label",
        "anomaly_score",
        "fallback_anomaly_threshold",
        "annotated_image_path",
        "heatmap_path",
        "presence_mask_path",
        "presence_time_ms",
        "anomaly_time_ms",
        "total_time_ms",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_log_rows(tmp_path: Path) -> list[dict[str, str]]:
    return [
        {
            "run_id": "run_1",
            "timestamp": "2026-05-14T09:00:00",
            "image_name": "ok.png",
            "image_path": str(tmp_path / "ok.png"),
            "final_result": "OK",
            "presence_status": "PART_PRESENT",
            "foreground_ratio": "0.250000",
            "mean_diff": "12.000000",
            "largest_blob_area": "550.00",
            "changed_pixel_count": "1000",
            "zone_pixel_count": "4000",
            "anomaly_ran": "true",
            "anomaly_backend": "backend",
            "anomaly_pred_label": "False",
            "anomaly_score": "0.100000",
            "fallback_anomaly_threshold": "0.500000",
            "annotated_image_path": str(tmp_path / "ok_annotated.png"),
            "heatmap_path": "",
            "presence_mask_path": "",
            "presence_time_ms": "5.000",
            "anomaly_time_ms": "10.000",
            "total_time_ms": "15.000",
            "error_message": "",
        },
        {
            "run_id": "run_2",
            "timestamp": "2026-05-14T09:01:00",
            "image_name": "ng.png",
            "image_path": str(tmp_path / "ng.png"),
            "final_result": "NG",
            "presence_status": "PART_PRESENT",
            "foreground_ratio": "0.300000",
            "mean_diff": "20.000000",
            "largest_blob_area": "900.00",
            "changed_pixel_count": "1200",
            "zone_pixel_count": "4000",
            "anomaly_ran": "true",
            "anomaly_backend": "backend",
            "anomaly_pred_label": "True",
            "anomaly_score": "0.900000",
            "fallback_anomaly_threshold": "0.500000",
            "annotated_image_path": str(tmp_path / "ng_annotated.png"),
            "heatmap_path": str(tmp_path / "ng_heatmap.png"),
            "presence_mask_path": str(tmp_path / "ng_mask.png"),
            "presence_time_ms": "6.000",
            "anomaly_time_ms": "12.000",
            "total_time_ms": "18.000",
            "error_message": "defect found",
        },
    ]


def test_load_inspection_log_loads_valid_synthetic_log(tmp_path):
    log_path = tmp_path / "inspection_log.csv"
    write_synthetic_log(log_path, synthetic_log_rows(tmp_path))

    rows, fieldnames = load_inspection_log(log_path)

    assert len(rows) == 2
    assert "final_result" in fieldnames
    assert rows[0]["image_name"] == "ok.png"


def test_load_inspection_log_rejects_missing_required_columns(tmp_path):
    log_path = tmp_path / "not_an_inspection_log.csv"
    log_path.write_text("name,value\nfoo,bar\n", encoding="utf-8")

    try:
        load_inspection_log(log_path)
    except ValueError as exc:
        assert "missing required column" in str(exc)
    else:
        raise AssertionError("Expected unsupported CSV schema to be rejected.")


def test_log_load_worker_cancel_before_start_avoids_file_io(tmp_path):
    worker = LogLoadWorker("csv", tmp_path / "missing.csv")
    cancelled = []
    failed = []
    finished = []
    worker.cancelled.connect(cancelled.append)
    worker.failed.connect(failed.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.request_cancel()
    worker.run()

    assert cancelled == ["Log loading cancelled."]
    assert failed == []
    assert finished == [True]


def test_find_log_files_finds_known_project_logs(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    inspection_log = tmp_path / "inspection_log.csv"
    summary = nested / "summary.csv"
    other = nested / "other.csv"
    inspection_log.write_text("timestamp,final_result,image_name\n", encoding="utf-8")
    summary.write_text("timestamp,final_result,image_name\n", encoding="utf-8")
    other.write_text("timestamp,final_result,image_name\n", encoding="utf-8")

    found = find_log_files(tmp_path)

    assert inspection_log in found
    assert summary in found
    assert other not in found


def test_discover_job_history_logs_uses_supported_job_layout(tmp_path):
    camera_log = tmp_path / "camera" / "inspection_log.csv"
    image_log = tmp_path / "image" / "inspection_log.csv"
    folder_log_1 = tmp_path / "folder" / "run_20260515_100000_000001" / "summary.csv"
    folder_log_2 = tmp_path / "folder" / "run_20260515_100001_000002" / "summary.csv"
    unrelated = tmp_path / "other" / "inspection_log.csv"
    for path in (camera_log, image_log, folder_log_1, folder_log_2, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("timestamp,final_result,image_name\n", encoding="utf-8")

    discovered = discover_job_history_logs(tmp_path)

    assert discovered == [
        (camera_log, "Camera"),
        (image_log, "Image"),
        (folder_log_1, "Folder"),
        (folder_log_2, "Folder"),
    ]


def test_infer_log_mode_from_known_paths():
    assert infer_log_mode("outputs/transistor/camera/inspection_log.csv") == "Camera"
    assert infer_log_mode("outputs/transistor/image/inspection_log.csv") == "Image"
    assert infer_log_mode("outputs/transistor/folder/run_20260515_100000_000001/summary.csv") == "Folder"
    assert infer_log_mode("outputs/random/summary.csv") == "Log"


def test_rows_with_log_metadata_adds_mode_and_source(tmp_path):
    source = tmp_path / "camera" / "inspection_log.csv"
    rows = rows_with_log_metadata([{"timestamp": "2026-05-15T10:00:00", "final_result": "OK"}], source, "Camera")

    assert rows[0][LOG_MODE_KEY] == "Camera"
    assert rows[0][LOG_SOURCE_KEY] == str(source)


def test_rows_with_log_metadata_prefers_explicit_mode_over_path_inference(tmp_path):
    source = tmp_path / "camera" / "inspection_log.csv"
    rows = rows_with_log_metadata(
        [{"timestamp": "2026-05-15T10:00:00", "final_result": "OK", "inspection_mode": "folder"}],
        source,
        "Camera",
    )

    assert rows[0][LOG_MODE_KEY] == "Folder"


def test_mode_label_normalizes_csv_mode_values():
    assert mode_label("camera") == "Camera"
    assert mode_label("image") == "Image"
    assert mode_label("folder") == "Folder"
    assert mode_label("cli_image") == ""


def test_load_job_history_rows_combines_modes_sources_and_sorts(tmp_path):
    camera_log = tmp_path / "camera" / "inspection_log.csv"
    image_log = tmp_path / "image" / "inspection_log.csv"
    folder_log = tmp_path / "folder" / "run_20260515_100000_000001" / "summary.csv"
    write_synthetic_log(camera_log, [{**synthetic_log_rows(tmp_path)[0], "timestamp": "2026-05-15T10:00:00"}])
    write_synthetic_log(image_log, [{**synthetic_log_rows(tmp_path)[0], "timestamp": "2026-05-15T10:02:00"}])
    write_synthetic_log(folder_log, [{**synthetic_log_rows(tmp_path)[0], "timestamp": "2026-05-15T10:01:00"}])

    rows, sources = load_job_history_rows(tmp_path)

    assert sources == [camera_log, image_log, folder_log]
    assert [row[LOG_MODE_KEY] for row in rows] == ["Image", "Folder", "Camera"]
    assert [row["timestamp"] for row in rows] == [
        "2026-05-15T10:02:00",
        "2026-05-15T10:01:00",
        "2026-05-15T10:00:00",
    ]
    assert {row[LOG_SOURCE_KEY] for row in rows} == {str(camera_log), str(image_log), str(folder_log)}


def test_filter_log_rows_filters_by_result_search_and_errors(tmp_path):
    rows = synthetic_log_rows(tmp_path)

    assert [row["image_name"] for row in filter_log_rows(rows, "NG")] == ["ng.png"]
    assert [row["image_name"] for row in filter_log_rows(rows, "All", "ok")] == ["ok.png"]
    assert [row["image_name"] for row in filter_log_rows(rows, "All", "", True)] == ["ng.png"]


def test_filter_log_rows_filters_by_mode_and_composes_with_result_search_errors(tmp_path):
    rows = rows_with_log_metadata(synthetic_log_rows(tmp_path), tmp_path / "camera" / "inspection_log.csv", "Camera")
    rows.extend(rows_with_log_metadata([{**synthetic_log_rows(tmp_path)[1], "image_name": "folder_ng.png"}], tmp_path / "folder" / "run_1" / "summary.csv", "Folder"))

    assert [row["image_name"] for row in filter_log_rows(rows, mode_filter="Folder")] == ["folder_ng.png"]
    assert [row["image_name"] for row in filter_log_rows(rows, "NG", "folder", True, "Folder")] == ["folder_ng.png"]


def test_row_image_name_falls_back_to_image_path():
    assert row_image_name({"image_path": "outputs/run/image.png"}) == "image.png"


def test_default_logs_output_dir_uses_app_state():
    assert default_logs_output_dir(AppState(logs_root_dir=Path("outputs/current"))) == Path("outputs/current")


def test_default_logs_output_dir_falls_back_to_job_root():
    assert default_logs_output_dir(AppState(inspection_job_name="Transistor")) == Path("outputs/transistor")


def test_logs_default_output_tracks_job_until_manual_override():
    app = QApplication.instance() or QApplication([])
    state = AppState()
    page = LogsPage(state)

    state.set_inspection_job_name("Transistor")
    page.refresh_from_state()

    assert normalized_path_text(page.output_folder_edit.text()) == "outputs/transistor"

    state.logs_root_dir = Path("outputs/manual_logs")
    state.set_inspection_job_name("Metal Surface")
    page.refresh_from_state()

    assert normalized_path_text(page.output_folder_edit.text()) == "outputs/manual_logs"


def test_logs_text_edit_marks_manual_root_override():
    app = QApplication.instance() or QApplication([])
    state = AppState(inspection_job_name="Transistor")
    page = LogsPage(state)

    page.output_folder_edit.setText("outputs/custom_logs")
    page._mark_logs_root_override("outputs/custom_logs")
    state.set_inspection_job_name("Metal Surface")
    page.refresh_from_state()

    assert state.logs_root_dir == Path("outputs/custom_logs")
    assert normalized_path_text(page.output_folder_edit.text()) == "outputs/custom_logs"


def test_logs_page_constructs_with_expected_controls():
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState(logs_root_dir=Path("outputs/current")))
    dimensions = theme_dimensions()

    assert page.findChild(QSplitter, "logsReviewSplitter") is not None
    assert page.findChild(QWidget, "logsReviewToolbar") is not None
    assert page.findChild(QWidget, "logsRecordsPane") is not None
    assert page.findChild(QWidget, "logsReviewPane") is not None
    assert page.findChild(QGroupBox, "logsSourceControls") is not None
    assert page.findChild(QGroupBox, "logsFilterControls") is not None
    assert page.findChild(QWidget, "selectedRecordReview") is not None
    assert page.findChild(QScrollArea, "logsSelectedRecordScroll") is None
    assert page.findChild(QWidget, "logsSelectedRecordContext") is not None
    assert page.findChild(QGroupBox, "logsArtifactReview") is None
    assert page.findChild(QWidget, "logsMoreDetails") is not None
    assert page.findChild(QWidget, "logsSelectedRecordSummary") is not None
    assert page.findChild(QWidget, "logsReviewActions") is not None
    assert isinstance(page.feedback_label, StatusBanner)
    assert isinstance(page.review_feedback_label, StatusBanner)
    assert isinstance(page.records_table, QTableView)
    assert page.result_filter_combo.count() == 5
    assert page.mode_filter_combo.count() == 4
    assert page.load_job_history_button.text() == "Load Job History"
    assert page.view_images_button.text() == "View Log Images"
    assert normalized_path_text(page.output_folder_edit.text()) == "outputs/current"
    assert page.source_summary_label.text() == "No logs loaded."
    assert page.empty_review_label.isHidden() is False
    assert page.selected_record_widget.isHidden() is True
    assert page.summary_grid_widget.isHidden() is True
    assert page.more_details_widget.isHidden() is True
    assert page.view_images_button.isEnabled() is False
    assert dimensions.logs_splitter_records_width > dimensions.logs_splitter_review_width
    assert dimensions.logs_splitter_records_stretch > dimensions.logs_splitter_review_stretch
    assert dimensions.logs_splitter_review_width >= 480
    assert dimensions.logs_splitter_records_stretch == 7
    assert dimensions.logs_splitter_review_stretch == 3


def test_logs_records_table_uses_balanced_column_resize_modes():
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    header = page.records_table.horizontalHeader()

    assert page.records_proxy_model.columnCount() == 7
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(4) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(5) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(6) == QHeaderView.ResizeMode.Fixed


def test_logs_records_table_uses_theme_styles():
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    stylesheet = build_app_stylesheet()

    assert "QTableView#recordsTable" in stylesheet
    assert "selection-background-color" in stylesheet
    assert "QSplitter#referenceCaptureWorkspace" in stylesheet
    assert "QSplitter#zoneEditorWorkspace" in stylesheet
    assert "QWidget#logsSelectedRecordSummary" in stylesheet
    assert "QWidget#logsRecordsPane" in stylesheet
    assert "QWidget#logsReviewPane" in stylesheet
    assert "QWidget#logsReviewActions" in stylesheet
    assert "QWidget#logsReviewToolbar" in stylesheet
    assert "QFrame#topNavShell[compact=\"true\"]" in stylesheet
    assert page.records_table.viewport().styleSheet() == table_viewport_stylesheet()


def test_logs_records_table_populates_with_column_alignment(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    page._rows = rows_with_log_metadata(synthetic_log_rows(tmp_path), tmp_path / "camera" / "inspection_log.csv", "Camera")

    page._set_model_rows()

    model = page.records_proxy_model
    assert model.rowCount() == 2
    assert model.data(model.index(0, 1)) == "Camera"
    assert model.data(model.index(0, 2)) == "OK"
    assert model.data(model.index(0, 3)) == "ok.png"
    assert model.data(model.index(0, 4)) == "Part present"
    assert model.data(model.index(0, 5)) == "0.1"
    assert model.data(model.index(0, 6)) == "15 ms"
    assert model.data(model.index(0, 2), Qt.ItemDataRole.TextAlignmentRole) == Qt.AlignmentFlag.AlignCenter
    assert model.data(model.index(0, 5), Qt.ItemDataRole.TextAlignmentRole) == (
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )


def test_logs_page_updates_details_and_hides_empty_error(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    source = tmp_path / "camera" / "inspection_log.csv"
    row = rows_with_log_metadata(
        [{**synthetic_log_rows(tmp_path)[0], "inspection_job": "Transistor", "inspection_job_slug": "transistor"}],
        source,
        "Camera",
    )[0]

    page._update_selected_record(row)

    assert page.empty_review_label.isHidden() is True
    assert page.selected_record_widget.isHidden() is False
    assert page.summary_grid_widget.isHidden() is False
    assert page.more_details_widget.isHidden() is False
    assert page.record_result_badge.text() == "OK"
    assert page.record_result_badge.property("result") == "OK"
    assert page._detail_widgets[LOG_MODE_KEY].text() == "Camera"
    assert page._detail_widgets["inspection_job"].text() == "Transistor | Camera"
    assert page._detail_widgets["inspection_job_slug"].text() == "transistor"
    assert page._detail_widgets["presence_status"].text() == "Part present"
    assert page._detail_widgets["anomaly_pred_label"].text() == "No anomaly"
    assert page._detail_widgets["anomaly_score"].text() == "0.1"
    assert page._detail_widgets["total_time_ms"].text() == "15 ms"
    assert page._detail_widgets["foreground_ratio"].text() == "25.0%"
    assert page._detail_widgets["largest_blob_area"].text() == "550 px^2"
    assert page._detail_widgets[LOG_SOURCE_KEY].toPlainText() == str(source)
    assert page._error_label.isHidden() is True
    assert page._error_text.isHidden() is True
    assert page.view_images_button.isEnabled() is True


def test_logs_image_viewer_is_created_only_when_requested(tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "large_log_artifact.png"
    cv2.imwrite(str(image_path), np.zeros((1600, 2400, 3), dtype=np.uint8))
    page = LogsPage(AppState())
    row = rows_with_log_metadata(
        [
            {
                "timestamp": "2026-05-15T10:00:00",
                "final_result": "OK",
                "image_name": "large_log_artifact.png",
                "annotated_image_path": str(image_path),
                "heatmap_path": str(image_path),
                "presence_mask_path": str(image_path),
            }
        ],
        tmp_path / "camera" / "inspection_log.csv",
        "Camera",
    )[0]

    page._rows = [row]
    page._set_model_rows()
    assert page._log_image_dialog is None
    page.open_selected_log_images()
    app.processEvents()

    assert page._log_image_dialog is not None
    assert page._log_image_dialog.tabs.count() == 4
    assert page._log_image_dialog.annotated_preview._pixmap_item is not None
    page._log_image_dialog.close()


def test_logs_more_details_default_collapsed_and_expands_after_selection(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    row = rows_with_log_metadata([synthetic_log_rows(tmp_path)[0]], tmp_path / "image" / "inspection_log.csv", "Image")[0]

    page._update_selected_record(row)

    assert page.more_details_button.isChecked() is False
    assert page.more_details_scroll.isHidden() is True

    page.more_details_button.setChecked(True)

    assert page.more_details_scroll.isHidden() is False


def test_logs_details_remain_in_the_compact_review_pane(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    row = rows_with_log_metadata([synthetic_log_rows(tmp_path)[1]], tmp_path / "folder" / "run_1" / "summary.csv", "Folder")[0]
    page._update_selected_record(row)
    page.more_details_button.setChecked(True)
    app.processEvents()

    assert page.more_details_widget.parent().objectName() == "selectedRecordReview"


def test_logs_empty_state_replaces_blank_detail_grid():
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())

    page._clear_details_and_previews()

    assert "Select a record" in page.empty_review_label.text()
    assert page.empty_review_label.isHidden() is False
    assert page.selected_record_widget.isHidden() is True
    assert page.summary_grid_widget.isHidden() is True
    assert page.more_details_widget.isHidden() is True
    assert page.record_result_badge.text() == "No result"
    assert page.view_images_button.isEnabled() is False


def test_logs_page_shows_error_when_present(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    row = rows_with_log_metadata([synthetic_log_rows(tmp_path)[1]], tmp_path / "folder" / "run_1" / "summary.csv", "Folder")[0]

    page._update_selected_record(row)

    assert page._error_label.isHidden() is False
    assert page._error_text.isHidden() is False


def test_logs_page_single_csv_load_infers_mode_and_keeps_workflow(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    log_path = tmp_path / "camera" / "inspection_log.csv"
    write_synthetic_log(log_path, [synthetic_log_rows(tmp_path)[0]])
    page.csv_path_edit.setText(str(log_path))

    page.load_current_csv()
    finish_log_operation(page, app)

    assert len(page._rows) == 1
    assert page._rows[0][LOG_MODE_KEY] == "Camera"
    assert page._rows[0][LOG_SOURCE_KEY] == str(log_path)
    assert page.records_proxy_model.rowCount() == 1


def test_logs_page_load_job_history_combines_records(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    camera_log = tmp_path / "camera" / "inspection_log.csv"
    image_log = tmp_path / "image" / "inspection_log.csv"
    folder_log = tmp_path / "folder" / "run_20260515_100000_000001" / "summary.csv"
    write_synthetic_log(camera_log, [{**synthetic_log_rows(tmp_path)[0], "timestamp": "2026-05-15T10:00:00"}])
    write_synthetic_log(image_log, [{**synthetic_log_rows(tmp_path)[0], "timestamp": "2026-05-15T10:02:00"}])
    write_synthetic_log(folder_log, [{**synthetic_log_rows(tmp_path)[1], "timestamp": "2026-05-15T10:01:00"}])
    page.output_folder_edit.setText(str(tmp_path))

    page.load_job_history()
    finish_log_operation(page, app)

    assert len(page._rows) == 3
    assert [row[LOG_MODE_KEY] for row in page._rows] == ["Image", "Folder", "Camera"]
    assert page.records_proxy_model.rowCount() == 3


def test_logs_page_mode_filter_applies_to_loaded_records(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LogsPage(AppState())
    page._rows = rows_with_log_metadata([synthetic_log_rows(tmp_path)[0]], tmp_path / "camera" / "inspection_log.csv", "Camera")
    page._rows.extend(rows_with_log_metadata([synthetic_log_rows(tmp_path)[1]], tmp_path / "folder" / "run_1" / "summary.csv", "Folder"))

    page.mode_filter_combo.setCurrentText("Folder")
    page._set_model_rows()

    assert page.records_proxy_model.rowCount() == 1
    assert page._current_row()[LOG_MODE_KEY] == "Folder"
