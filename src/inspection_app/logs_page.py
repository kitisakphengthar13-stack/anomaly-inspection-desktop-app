from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QHeaderView,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inspection_app.formatting import format_log_value
from inspection_app.job_paths import default_logs_root_dir
from inspection_app.state import AppState
from inspection_app.theme import (
    group_content_margins,
    group_control_margins,
    page_margins,
    table_viewport_stylesheet,
    theme_dimensions,
    theme_spacing,
    zero_margins,
)
from inspection_app.ui_locale import apply_app_locale
from inspection_app.ui_components import ResultBadge, ResultImageTabs, ScrollablePane, StatusBanner, set_button_icon, set_button_role

StatusCallback = Callable[[str], None]

KNOWN_LOG_FILENAMES = {"inspection_log.csv", "summary.csv"}
REQUIRED_LOG_COLUMNS = {"timestamp", "final_result"}
LOG_MODE_KEY = "__mode"
LOG_SOURCE_KEY = "__source_log_path"
TABLE_COLUMNS = (
    ("timestamp", "Timestamp"),
    (LOG_MODE_KEY, "Mode"),
    ("final_result", "Result"),
    ("image_name", "Image Name"),
    ("presence_status", "Presence"),
    ("anomaly_score", "Score"),
    ("total_time_ms", "Inspection Time"),
)
DETAIL_FIELDS = (
    ("Mode", LOG_MODE_KEY),
    ("Inspection Job", "inspection_job"),
    ("Job Slug", "inspection_job_slug"),
    ("Timestamp", "timestamp"),
    ("Run ID", "run_id"),
    ("Image Name", "image_name"),
    ("Image Path", "image_path"),
    ("Final Result", "final_result"),
    ("Presence", "presence_status"),
    ("Foreground", "foreground_ratio"),
    ("Mean pixel difference", "mean_diff"),
    ("Blob area", "largest_blob_area"),
    ("Changed pixels", "changed_pixel_count"),
    ("Zone pixels", "zone_pixel_count"),
    ("Anomaly Ran", "anomaly_ran"),
    ("Backend", "anomaly_backend"),
    ("Anomaly decision", "anomaly_pred_label"),
    ("Score", "anomaly_score"),
    ("Fallback threshold", "fallback_anomaly_threshold"),
    ("Presence check time", "presence_time_ms"),
    ("Anomaly inference time", "anomaly_time_ms"),
    ("Inspection time", "total_time_ms"),
)
SUMMARY_FIELDS = (
    ("Mode", LOG_MODE_KEY),
    ("Presence", "presence_status"),
    ("Score", "anomaly_score"),
    ("Inspection time", "total_time_ms"),
)
PRIMARY_REVIEW_KEYS = {"final_result", "timestamp", "image_name", "inspection_job"}
SUMMARY_KEYS = {key for _, key in SUMMARY_FIELDS} | PRIMARY_REVIEW_KEYS
MORE_DETAIL_FIELDS = tuple((label, key) for label, key in DETAIL_FIELDS if key not in SUMMARY_KEYS)


def default_logs_output_dir(state: AppState) -> Path:
    return state.logs_root_dir or default_logs_root_dir(state.inspection_job_slug)


def load_inspection_log(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"CSV log file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV log path is not a file: {csv_path}")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("CSV log has no header row.")
            fieldnames = [field.strip() for field in reader.fieldnames]
            _validate_log_columns(fieldnames)
            rows = [{key: (value or "") for key, value in row.items()} for row in reader]
    except csv.Error as exc:
        raise ValueError(f"Could not parse CSV log: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not read CSV log as UTF-8: {exc}") from exc

    if not rows:
        raise ValueError("CSV log is empty.")
    return rows, fieldnames


def _validate_log_columns(fieldnames: list[str]) -> None:
    fields = set(fieldnames)
    missing = sorted(REQUIRED_LOG_COLUMNS - fields)
    if missing:
        raise ValueError(f"CSV log is missing required column(s): {', '.join(missing)}")
    if "image_name" not in fields and "image_path" not in fields:
        raise ValueError("CSV log must include image_name or image_path.")


def find_log_files(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        raise ValueError(f"Output folder does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Output path is not a folder: {root}")
    return sorted(path for path in root.rglob("*.csv") if path.name in KNOWN_LOG_FILENAMES)


def discover_job_history_logs(job_root: str | Path) -> list[tuple[Path, str]]:
    root = Path(job_root)
    if not root.exists():
        raise ValueError(f"Job history root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Job history root is not a folder: {root}")

    candidates = (
        (root / "camera" / "inspection_log.csv", "Camera"),
        (root / "image" / "inspection_log.csv", "Image"),
    )
    logs = [(path, mode) for path, mode in candidates if path.is_file()]
    folder_root = root / "folder"
    if folder_root.is_dir():
        logs.extend((path, "Folder") for path in sorted(folder_root.glob("run_*/summary.csv")) if path.is_file())
    return logs


def infer_log_mode(path: str | Path) -> str:
    log_path = Path(path)
    parent = log_path.parent.name.lower()
    if log_path.name == "inspection_log.csv" and parent == "camera":
        return "Camera"
    if log_path.name == "inspection_log.csv" and parent == "image":
        return "Image"
    if log_path.name == "summary.csv" and log_path.parent.name.startswith("run_") and log_path.parent.parent.name.lower() == "folder":
        return "Folder"
    return "Log"


def rows_with_log_metadata(rows: list[dict[str, str]], source_log_path: str | Path, mode: str) -> list[dict[str, str]]:
    source = str(Path(source_log_path))
    enriched: list[dict[str, str]] = []
    for row in rows:
        explicit_mode = mode_label(row.get("inspection_mode", ""))
        enriched.append({**row, LOG_MODE_KEY: explicit_mode or mode, LOG_SOURCE_KEY: source})
    return enriched


def mode_label(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "camera":
        return "Camera"
    if normalized == "image":
        return "Image"
    if normalized == "folder":
        return "Folder"
    return ""


def load_job_history_rows(job_root: str | Path) -> tuple[list[dict[str, str]], list[Path]]:
    discovered = discover_job_history_logs(job_root)
    if not discovered:
        raise ValueError(f"No camera, image, or folder-run logs found under job root: {Path(job_root)}")

    combined: list[dict[str, str]] = []
    source_paths: list[Path] = []
    for path, mode in discovered:
        rows, _ = load_inspection_log(path)
        combined.extend(rows_with_log_metadata(rows, path, mode))
        source_paths.append(path)
    combined.sort(key=_timestamp_sort_key, reverse=True)
    return combined, source_paths


def _timestamp_sort_key(row: dict[str, str]) -> tuple[int, str]:
    value = row.get("timestamp", "").strip()
    if not value:
        return (0, "")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return (0, "")
    return (1, value)


def filter_log_rows(
    rows: list[dict[str, str]],
    result_filter: str = "All",
    search_text: str = "",
    errors_only: bool = False,
    mode_filter: str = "All Modes",
) -> list[dict[str, str]]:
    query = search_text.strip().lower()
    filtered: list[dict[str, str]] = []
    for row in rows:
        if mode_filter != "All Modes" and row.get(LOG_MODE_KEY, "Log") != mode_filter:
            continue
        if result_filter != "All" and row.get("final_result", "") != result_filter:
            continue
        if errors_only and not row.get("error_message", "").strip():
            continue
        if query:
            haystack = f"{row_image_name(row)} {row.get('image_path', '')}".lower()
            if query not in haystack:
                continue
        filtered.append(row)
    return filtered


def row_image_name(row: dict[str, str]) -> str:
    name = row.get("image_name", "").strip()
    if name:
        return name
    path = row.get("image_path", "").strip()
    return Path(path).name if path else ""


def _table_alignment_for_column(key: str) -> Qt.AlignmentFlag:
    if key == "image_name":
        return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    if key in {"anomaly_score", "total_time_ms"}:
        return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    return Qt.AlignmentFlag.AlignCenter


class LogsPage(QWidget):
    def __init__(self, state: AppState, status_callback: StatusCallback | None = None) -> None:
        super().__init__()
        self.state = state
        self._status_callback = status_callback
        self._rows: list[dict[str, str]] = []
        self._filtered_rows: list[dict[str, str]] = []
        self._current_csv_path: Path | None = None
        self._current_source_paths: list[Path] = []
        self._detail_widgets: dict[str, QLabel | QPlainTextEdit] = {}
        self._summary_widgets: dict[str, QLabel] = {}
        self._error_label: QLabel | None = None
        self._error_text: QPlainTextEdit | None = None
        self._has_selected_record = False

        self._build_ui()
        apply_app_locale(self)
        self.refresh_from_state()
        self._set_feedback("Load history to review results.", ok=True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        spacing = theme_spacing()
        layout.setContentsMargins(*page_margins())
        layout.setSpacing(spacing.operations_page_gap)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.feedback_label = StatusBanner()
        self.review_feedback_label = StatusBanner()

        layout.addWidget(self._review_toolbar())
        layout.addWidget(self._review_console(), 1)

    def _review_toolbar(self) -> QWidget:
        toolbar = QWidget()
        toolbar.setObjectName("logsReviewToolbar")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().section_gap)
        layout.addWidget(self._source_group(), 3)
        layout.addWidget(self._filters_group(), 2)
        return toolbar

    def _review_console(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("logsReviewSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._split_pane(self._records_group(), "logsRecordsPane", right_gutter=True))
        splitter.addWidget(self._split_pane(self._selected_record_review_group(), "logsReviewPane", left_gutter=True))
        dimensions = theme_dimensions()
        splitter.setStretchFactor(0, dimensions.logs_splitter_records_stretch)
        splitter.setStretchFactor(1, dimensions.logs_splitter_review_stretch)
        splitter.setSizes([dimensions.logs_splitter_records_width, dimensions.logs_splitter_review_width])
        return splitter

    def _split_pane(self, child: QWidget, object_name: str, *, left_gutter: bool = False, right_gutter: bool = False) -> QWidget:
        pane = QWidget()
        pane.setObjectName(object_name)
        spacing = theme_spacing()
        gutter = spacing.control_gap
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(gutter if left_gutter else 0, 0, gutter if right_gutter else 0, 0)
        layout.setSpacing(0)
        layout.addWidget(child, 1)
        return pane

    def _selected_record_review_group(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("selectedRecordReview")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().logs_review_gap)
        context_scroll = ScrollablePane(self._selected_record_context(), object_name="logsSelectedRecordScroll")
        context_scroll.setMaximumHeight(theme_dimensions().logs_selected_record_context_max_height)
        layout.addWidget(context_scroll)
        layout.addWidget(self._artifact_group(), 1)
        layout.addWidget(self._more_details_group())
        return widget

    def _selected_record_context(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("logsSelectedRecordContext")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().logs_review_gap)
        layout.addWidget(self._summary_group())
        layout.addWidget(self._review_actions_group())
        layout.addWidget(self.review_feedback_label)
        layout.addStretch(1)
        return widget

    def _source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        group.setObjectName("logsSourceControls")
        layout = QGridLayout(group)
        spacing = theme_spacing()
        layout.setContentsMargins(*group_control_margins())
        layout.setHorizontalSpacing(spacing.logs_filter_gap)
        layout.setVerticalSpacing(max(4, spacing.logs_filter_gap - 2))

        self.csv_path_edit = QLineEdit()
        browse_csv = QPushButton("Browse CSV Log...")
        browse_csv.clicked.connect(self._browse_csv_log)
        load_csv = QPushButton("Load Log")
        load_csv.clicked.connect(self.load_current_csv)
        layout.addWidget(QLabel("CSV"), 0, 0)
        layout.addWidget(self.csv_path_edit, 0, 1)
        layout.addWidget(browse_csv, 0, 2)
        layout.addWidget(load_csv, 0, 3)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.textEdited.connect(self._mark_logs_root_override)
        choose_folder = QPushButton("Choose Output Folder...")
        choose_folder.clicked.connect(self._choose_output_folder)
        self.load_job_history_button = QPushButton("Load Job History")
        self.load_job_history_button.clicked.connect(self.load_job_history)
        layout.addWidget(QLabel("Job root"), 1, 0)
        layout.addWidget(self.output_folder_edit, 1, 1)
        layout.addWidget(choose_folder, 1, 2)
        layout.addWidget(self.load_job_history_button, 1, 3)

        self.detected_logs_combo = QComboBox()
        self.detected_logs_combo.setEnabled(False)
        find_logs = QPushButton("Find Logs")
        find_logs.clicked.connect(self.find_logs_in_output_folder)
        self.load_detected_button = QPushButton("Load Selected Log")
        self.load_detected_button.setEnabled(False)
        self.load_detected_button.clicked.connect(self.load_selected_detected_log)
        self.source_summary_label = QLabel("No logs loaded.")
        self.source_summary_label.setObjectName("logsSourceSummary")
        self.source_summary_label.setWordWrap(False)
        layout.addWidget(self.source_summary_label, 2, 1)
        layout.addWidget(self.detected_logs_combo, 2, 2)
        layout.addWidget(find_logs, 2, 3)
        layout.addWidget(self.load_detected_button, 2, 4)
        layout.addWidget(self.feedback_label, 3, 0, 1, 5)
        layout.setColumnStretch(1, 1)
        return group

    def _filters_group(self) -> QGroupBox:
        group = QGroupBox("Filter")
        group.setObjectName("logsFilterControls")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(*group_control_margins())
        layout.setSpacing(theme_spacing().logs_filter_gap)

        self.result_filter_combo = QComboBox()
        self.result_filter_combo.addItems(["All", "OK", "NG", "NO_PART", "ERROR"])
        self.result_filter_combo.currentTextChanged.connect(lambda _text: self.apply_filters())

        self.mode_filter_combo = QComboBox()
        self.mode_filter_combo.addItems(["All Modes", "Camera", "Image", "Folder"])
        self.mode_filter_combo.currentTextChanged.connect(lambda _text: self.apply_filters())

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search image name or path")
        self.search_edit.textChanged.connect(lambda _text: self.apply_filters())

        self.errors_only_checkbox = QCheckBox("Errors only")
        self.errors_only_checkbox.stateChanged.connect(lambda _state: self.apply_filters())

        layout.addWidget(QLabel("Result"))
        layout.addWidget(self.result_filter_combo)
        layout.addSpacing(theme_spacing().compact_control_gap)
        layout.addWidget(QLabel("Mode"))
        layout.addWidget(self.mode_filter_combo)
        layout.addSpacing(theme_spacing().compact_control_gap)
        layout.addWidget(QLabel("Search"))
        layout.addWidget(self.search_edit, 1)
        layout.addWidget(self.errors_only_checkbox)
        return group

    def _records_group(self) -> QGroupBox:
        group = QGroupBox("Inspection Records")
        group.setMinimumHeight(theme_dimensions().logs_records_min_height)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(*group_content_margins())

        self.records_table = QTableWidget(0, len(TABLE_COLUMNS))
        self.records_table.setHorizontalHeaderLabels([label for _, label in TABLE_COLUMNS])
        self.records_table.setObjectName("recordsTable")
        self.records_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.records_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.records_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.records_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.records_table.setAlternatingRowColors(True)
        self.records_table.setShowGrid(False)
        self.records_table.setWordWrap(False)
        self.records_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.records_table.viewport().setStyleSheet(table_viewport_stylesheet())
        self.records_table.verticalHeader().setVisible(False)
        self.records_table.verticalHeader().setDefaultSectionSize(theme_dimensions().logs_table_row_height)
        self._configure_records_table_header()
        self.records_table.itemSelectionChanged.connect(self._on_record_selection_changed)
        layout.addWidget(self.records_table)
        return group

    def _configure_records_table_header(self) -> None:
        header = self.records_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        dimensions = theme_dimensions()
        header.setMinimumSectionSize(dimensions.logs_table_header_min_section_width)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.records_table.setColumnWidth(0, dimensions.logs_timestamp_column_width)
        self.records_table.setColumnWidth(1, dimensions.logs_mode_column_width)
        self.records_table.setColumnWidth(2, dimensions.logs_result_column_width)
        self.records_table.setColumnWidth(4, dimensions.logs_presence_column_width)
        self.records_table.setColumnWidth(5, dimensions.logs_score_column_width)
        self.records_table.setColumnWidth(6, dimensions.logs_total_time_column_width)

    def _summary_group(self) -> QGroupBox:
        group = QGroupBox("Current Review")
        layout = QVBoxLayout(group)
        spacing = theme_spacing()
        layout.setContentsMargins(*group_control_margins())
        layout.setSpacing(spacing.compact_control_gap)

        self.empty_review_label = QLabel("No record selected\nSelect a record to review result images.")
        self.empty_review_label.setObjectName("logsEmptyReview")
        self.empty_review_label.setWordWrap(True)
        layout.addWidget(self.empty_review_label)

        self.selected_record_widget = QWidget()
        self.selected_record_widget.setObjectName("logsSelectedRecordSummary")
        selected_layout = QVBoxLayout(self.selected_record_widget)
        selected_layout.setContentsMargins(*zero_margins())
        selected_layout.setSpacing(spacing.compact_control_gap)

        self.record_result_badge = ResultBadge()
        selected_layout.addWidget(self.record_result_badge)
        self._detail_widgets["final_result"] = self.record_result_badge

        self.record_identity_label = QLabel("")
        self.record_identity_label.setObjectName("logsRecordIdentity")
        self.record_identity_label.setWordWrap(True)
        self.record_identity_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        selected_layout.addWidget(self.record_identity_label)
        self._detail_widgets["image_name"] = self.record_identity_label

        self.record_time_label = QLabel("")
        self.record_time_label.setObjectName("logsRecordTime")
        self.record_time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        selected_layout.addWidget(self.record_time_label)
        self._detail_widgets["timestamp"] = self.record_time_label

        self.record_context_label = QLabel("")
        self.record_context_label.setObjectName("logsRecordContext")
        self.record_context_label.setWordWrap(True)
        self.record_context_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        selected_layout.addWidget(self.record_context_label)
        self._detail_widgets["inspection_job"] = self.record_context_label

        self.summary_grid_widget = QWidget()
        self.summary_grid_widget.setObjectName("logsSummaryGrid")
        grid = QGridLayout()
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)
        grid.setContentsMargins(*zero_margins())
        for index, (label_text, key) in enumerate(SUMMARY_FIELDS):
            row = index // 2
            column = 0 if index % 2 == 0 else 2
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value = QLabel("")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._summary_widgets[key] = value
            self._detail_widgets[key] = value
            grid.addWidget(label, row, column)
            grid.addWidget(value, row, column + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.summary_grid_widget.setLayout(grid)
        self.summary_grid_widget.setVisible(False)
        selected_layout.addWidget(self.summary_grid_widget)
        self.selected_record_widget.setVisible(False)
        layout.addWidget(self.selected_record_widget)
        return group

    def _review_actions_group(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("logsReviewActions")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().control_gap)

        self.open_input_button = set_button_role(QPushButton("Open Input Image"), "secondary")
        set_button_icon(self.open_input_button, "open")
        self.open_input_button.clicked.connect(self.open_selected_input_image)
        self.open_annotated_button = set_button_role(QPushButton("Open Annotated Image"), "secondary")
        set_button_icon(self.open_annotated_button, "open")
        self.open_annotated_button.clicked.connect(self.open_selected_annotated_image)
        self.open_csv_button = set_button_role(QPushButton("Open CSV File"), "secondary")
        set_button_icon(self.open_csv_button, "open")
        self.open_csv_button.clicked.connect(self.open_csv_file)
        self.open_output_button = set_button_role(QPushButton("Open Output Folder"), "secondary")
        set_button_icon(self.open_output_button, "open")
        self.open_output_button.clicked.connect(self.open_output_folder)

        for button in (
            self.open_input_button,
            self.open_annotated_button,
            self.open_csv_button,
            self.open_output_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)
        self._set_review_actions_enabled(False)
        return widget

    def _more_details_group(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("logsMoreDetails")
        self.more_details_widget = widget
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().compact_control_gap)

        self.more_details_button = QPushButton("More Details")
        self.more_details_button.setObjectName("moreDetailsButton")
        self.more_details_button.setCheckable(True)
        self.more_details_button.toggled.connect(self._set_more_details_visible)
        layout.addWidget(self.more_details_button)

        details_scroll = QScrollArea()
        details_scroll.setObjectName("selectedRecordDetailsScroll")
        details_scroll.setWidgetResizable(True)
        details_scroll.setFrameShape(QFrame.Shape.NoFrame)
        details_scroll.setMaximumHeight(theme_dimensions().logs_details_scroll_max_height)
        details_scroll.setWidget(self._details_group())
        details_scroll.setVisible(False)
        self.more_details_scroll = details_scroll
        layout.addWidget(details_scroll)
        widget.setVisible(False)
        return widget

    def _details_group(self) -> QGroupBox:
        group = QGroupBox("Technical Details")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(*group_control_margins())
        layout.setSpacing(theme_spacing().compact_control_gap)

        grid = QGridLayout()
        spacing = theme_spacing()
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)
        for index, (label_text, key) in enumerate(MORE_DETAIL_FIELDS):
            row = index // 2
            column = 0 if index % 2 == 0 else 2
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value = QLabel("")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._detail_widgets[key] = value
            grid.addWidget(label, row, column)
            grid.addWidget(value, row, column + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        dimensions = theme_dimensions()
        for label_text, key, height in (
            ("Source Log File", LOG_SOURCE_KEY, dimensions.logs_source_text_height),
            ("Image Path", "image_path_long", dimensions.logs_image_path_text_height),
            ("Error Message", "error_message", dimensions.logs_error_text_height),
        ):
            label = QLabel(label_text)
            value = QPlainTextEdit()
            value.setReadOnly(True)
            value.setFixedHeight(height)
            value.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            self._detail_widgets[key] = value
            layout.addWidget(label)
            layout.addWidget(value)
            if key == "error_message":
                self._error_label = label
                self._error_text = value
        self._set_error_visible(False)
        return group

    def _set_more_details_visible(self, visible: bool) -> None:
        self.more_details_scroll.setVisible(visible)

    def _artifact_group(self) -> QGroupBox:
        group = QGroupBox("Artifacts")
        group.setObjectName("logsArtifactReview")
        group.setMinimumHeight(theme_dimensions().logs_artifact_min_height)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(*group_content_margins())
        layout.setSpacing(theme_spacing().logs_filter_gap)

        self.result_image_tabs = ResultImageTabs(
            annotated_placeholder="No annotated image selected.",
            heatmap_placeholder="No heatmap selected.",
            mask_placeholder="No presence mask selected.",
        )
        self.fit_image_button = self.result_image_tabs.fit_image_button
        self.tabs = self.result_image_tabs.tabs
        self.annotated_preview = self.result_image_tabs.annotated_preview
        self.heatmap_preview = self.result_image_tabs.heatmap_preview
        self.mask_preview = self.result_image_tabs.mask_preview
        layout.addWidget(self.result_image_tabs, 1)
        return group

    def refresh_from_state(self) -> None:
        if self.state.logs_root_dir is None:
            self.output_folder_edit.setText(str(default_logs_output_dir(self.state)))
        else:
            self.output_folder_edit.setText(str(self.state.logs_root_dir))

    def refresh_theme(self) -> None:
        self.records_table.viewport().setStyleSheet(table_viewport_stylesheet())

    def _browse_csv_log(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose inspection CSV log", self.csv_path_edit.text().strip(), "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_path_edit.setText(path)

    def _choose_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_folder_edit.text().strip())
        if path:
            self.output_folder_edit.setText(path)
            root_path = Path(path)
            self.state.logs_root_dir = root_path
            self.state.output_dir = root_path

    def _mark_logs_root_override(self, value: str) -> None:
        text = value.strip()
        self.state.logs_root_dir = Path(text) if text else None

    def find_logs_in_output_folder(self) -> None:
        try:
            output_dir = Path(self.output_folder_edit.text().strip())
            logs = find_log_files(output_dir)
        except Exception as exc:
            self._set_feedback(str(exc), ok=False)
            return

        self.detected_logs_combo.clear()
        for path in logs:
            self.detected_logs_combo.addItem(str(path), str(path))
        has_logs = bool(logs)
        self.detected_logs_combo.setEnabled(has_logs)
        self.load_detected_button.setEnabled(has_logs)
        if has_logs:
            self.csv_path_edit.setText(str(logs[0]))
            self.source_summary_label.setText(f"Detected {len(logs)} log file(s) under {output_dir}.")
            self._set_feedback(f"Found {len(logs)} log file(s). Select one and click Load Selected Log.", ok=True)
        else:
            self.source_summary_label.setText(f"No supported logs found under {output_dir}.")
            self._set_feedback(f"No inspection_log.csv or summary.csv files found under {output_dir}.", ok=False)

    def load_selected_detected_log(self) -> None:
        path = self.detected_logs_combo.currentData()
        if path:
            self.csv_path_edit.setText(str(path))
            self.load_current_csv()

    def load_current_csv(self) -> None:
        try:
            path = Path(self.csv_path_edit.text().strip())
            rows, _ = load_inspection_log(path)
        except Exception as exc:
            self._set_feedback(str(exc), ok=False)
            return

        self._current_csv_path = path
        self._current_source_paths = [path]
        self._rows = rows_with_log_metadata(rows, path, infer_log_mode(path))
        self.apply_filters()
        self.source_summary_label.setText(f"Loaded {len(rows)} record(s) from {path.name}.")
        self._set_feedback(f"Loaded {len(rows)} record(s) from {path}.", ok=True)

    def load_job_history(self) -> None:
        try:
            job_root = Path(self.output_folder_edit.text().strip())
            rows, source_paths = load_job_history_rows(job_root)
        except Exception as exc:
            self._set_feedback(str(exc), ok=False)
            return

        self._current_csv_path = None
        self._current_source_paths = source_paths
        self._rows = rows
        self.apply_filters()
        self.source_summary_label.setText(f"Loaded {len(rows)} record(s) from {len(source_paths)} job history log file(s).")
        self._set_feedback(
            f"Loaded {len(rows)} job history record(s) from {len(source_paths)} log file(s) under {job_root}.",
            ok=True,
        )

    def apply_filters(self) -> None:
        self._filtered_rows = filter_log_rows(
            self._rows,
            self.result_filter_combo.currentText(),
            self.search_edit.text(),
            self.errors_only_checkbox.isChecked(),
            self.mode_filter_combo.currentText(),
        )
        self._populate_table()
        if self._filtered_rows:
            self.records_table.selectRow(0)
        else:
            self._clear_details_and_previews()
            if self._rows:
                self._set_feedback("No records match the current filters.", ok=False)

    def _populate_table(self) -> None:
        self.records_table.setRowCount(len(self._filtered_rows))
        for row_index, row in enumerate(self._filtered_rows):
            for column_index, (key, _) in enumerate(TABLE_COLUMNS):
                raw_value = row_image_name(row) if key == "image_name" else row.get(key, "")
                value = raw_value if key in {"timestamp", LOG_MODE_KEY, "image_name"} else format_log_value(key, raw_value)
                item = QTableWidgetItem(value)
                item.setToolTip(raw_value if raw_value == value else f"{value} (raw: {raw_value})")
                item.setTextAlignment(_table_alignment_for_column(key))
                self.records_table.setItem(row_index, column_index, item)
        self.records_table.resizeRowsToContents()

    def _on_record_selection_changed(self) -> None:
        row_index = self.records_table.currentRow()
        if row_index < 0 or row_index >= len(self._filtered_rows):
            return
        self._update_selected_record(self._filtered_rows[row_index])

    def _update_selected_record(self, row: dict[str, str]) -> None:
        self._has_selected_record = True
        self.empty_review_label.setVisible(False)
        self.selected_record_widget.setVisible(True)
        self.summary_grid_widget.setVisible(True)
        self.more_details_widget.setVisible(True)
        self._set_review_actions_enabled(True)
        for _, key in DETAIL_FIELDS:
            widget = self._detail_widgets[key]
            if isinstance(widget, ResultBadge):
                final_result = row.get(key, "")
                widget.set_result(final_result)
            elif isinstance(widget, QLabel):
                if key == "image_name":
                    widget.setText(row_image_name(row) or "No image name")
                elif key == "timestamp":
                    widget.setText(f"Timestamp: {row.get(key, '')}")
                elif key == "inspection_job":
                    job_name = row.get(key, "").strip()
                    mode = row.get(LOG_MODE_KEY, "").strip()
                    widget.setText(" | ".join(part for part in (job_name, mode) if part))
                else:
                    widget.setText(format_log_value(key, row.get(key, "")))

        source_widget = self._detail_widgets[LOG_SOURCE_KEY]
        if isinstance(source_widget, QPlainTextEdit):
            source_widget.setPlainText(row.get(LOG_SOURCE_KEY, ""))
        image_widget = self._detail_widgets["image_path_long"]
        if isinstance(image_widget, QPlainTextEdit):
            image_widget.setPlainText(row.get("image_path", ""))
        error = row.get("error_message", "").strip()
        error_widget = self._detail_widgets["error_message"]
        if isinstance(error_widget, QPlainTextEdit):
            error_widget.setPlainText(error)
        self._set_error_visible(bool(error))

        self.result_image_tabs.set_artifacts(
            annotated_path=row.get("annotated_image_path"),
            heatmap_path=row.get("heatmap_path"),
            presence_mask_path=row.get("presence_mask_path"),
        )

    def _clear_details_and_previews(self) -> None:
        for widget in self._detail_widgets.values():
            if isinstance(widget, QPlainTextEdit):
                widget.clear()
            elif isinstance(widget, ResultBadge):
                widget.set_result("")
            elif isinstance(widget, QLabel):
                widget.setText("")
                if widget is self.record_result_badge:
                    widget.setText("No result")
                    widget.setProperty("result", "neutral")
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
        self._has_selected_record = False
        self.empty_review_label.setVisible(True)
        self.selected_record_widget.setVisible(False)
        self.summary_grid_widget.setVisible(False)
        self.more_details_widget.setVisible(False)
        self.more_details_button.setChecked(False)
        self._set_review_actions_enabled(False)
        self._set_error_visible(False)
        self.result_image_tabs.clear()

    def _set_error_visible(self, visible: bool) -> None:
        if self._error_label is not None:
            self._error_label.setVisible(visible)
        if self._error_text is not None:
            self._error_text.setVisible(visible)

    def _set_review_actions_enabled(self, has_selected_record: bool) -> None:
        self.open_input_button.setEnabled(has_selected_record)
        self.open_annotated_button.setEnabled(has_selected_record)

    def _fit_active_preview(self) -> None:
        self.result_image_tabs.fit_active_preview()

    def _refit_active_preview_if_needed(self) -> None:
        self.result_image_tabs.refit_active_preview_if_needed()

    def open_csv_file(self) -> None:
        self._open_path(self._current_csv_path, "No CSV log is loaded.")

    def open_output_folder(self) -> None:
        path_text = self.output_folder_edit.text().strip()
        self._open_path(Path(path_text) if path_text else None, "No output folder is selected.")

    def open_selected_input_image(self) -> None:
        row = self._current_row()
        path = row.get("image_path") if row else ""
        self._open_path(Path(path) if path else None, "Selected row has no input image path.")

    def open_selected_annotated_image(self) -> None:
        row = self._current_row()
        path = row.get("annotated_image_path") if row else ""
        self._open_path(Path(path) if path else None, "Selected row has no annotated image path.")

    def _current_row(self) -> dict[str, str] | None:
        row_index = self.records_table.currentRow()
        if row_index < 0 or row_index >= len(self._filtered_rows):
            return None
        return self._filtered_rows[row_index]

    def _open_path(self, path: Path | None, missing_message: str) -> None:
        if path is None:
            self._set_feedback(missing_message, ok=False, area="review")
            return
        if not path.exists():
            self._set_feedback(f"Path does not exist: {path}", ok=False, area="review")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self._set_feedback(f"Could not open path: {path}", ok=False, area="review")

    def _set_feedback(self, message: str, ok: bool, *, area: str = "source") -> None:
        target = self.review_feedback_label if area == "review" else self.feedback_label
        target.set_message(message, "success" if ok else "error")
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)
