from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inspection_app.formatting import format_final_result
from inspection_app.icons import (
    icon_pixmap,
    result_fallback_text,
    result_icon,
    state_fallback_text,
    state_icon,
)
from inspection_app.theme import active_theme, theme_dimensions, theme_spacing, zero_margins


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        spacing = theme_spacing()
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(spacing.control_gap // 2)

        title_label = QLabel(title)
        title_label.setObjectName("pageHeaderTitle")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("pageHeaderSubtitle")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)


class SectionPanel(QWidget):
    def __init__(self, title: str, subtitle: str | None = None, *, compact: bool = False) -> None:
        super().__init__()
        self.setObjectName("sectionPanel")
        if compact:
            self.setProperty("density", "compact")
        layout = QVBoxLayout(self)
        theme_gap = theme_spacing()
        padding = theme_gap.panel_padding if not compact else theme_gap.compact_panel_padding
        spacing = theme_gap.control_gap if not compact else theme_gap.compact_control_gap
        layout.setContentsMargins(padding, padding, padding, padding)
        layout.setSpacing(spacing)

        title_label = QLabel(title)
        title_label.setObjectName("sectionPanelTitle")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("sectionPanelSubtitle")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, spacing, 0, 0)
        self.content_layout.setSpacing(spacing)
        layout.addLayout(self.content_layout)


class IconLabel(QLabel):
    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.setObjectName("semanticIcon")
        self._icon_size = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_semantic_icon(self, icon_name: str, *, color: str | None = None, fallback_text: str = "") -> None:
        pixmap = icon_pixmap(state_icon(icon_name, color=color), self._icon_size)
        if pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(fallback_text or state_fallback_text(icon_name))
            return
        self.setText("")
        self.setPixmap(pixmap)

    def set_result_icon(self, result_name: str, *, color: str | None = None) -> None:
        pixmap = icon_pixmap(result_icon(result_name, color=color), self._icon_size)
        if pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(result_fallback_text(result_name))
            return
        self.setText("")
        self.setPixmap(pixmap)


class StatusBanner(QFrame):
    VALID_LEVELS = {"info", "success", "warning", "error"}
    STATE_LEVELS = {
        "idle": "info",
        "waiting": "info",
        "ready": "success",
        "blocked": "warning",
        "live": "info",
        "captured": "success",
        "processing": "info",
        "complete": "success",
        "error": "error",
        "saved": "success",
    }
    SPINNER_FRAMES = ("|", "/", "-", "\\")
    LIVE_FRAMES = ("●", "○")

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("statusBanner")
        self.setVisible(False)
        self._base_text = ""
        self._state = ""
        self._spinner_index = 0
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(10, 7, 10, 7)
        self._layout.setSpacing(theme_spacing().compact_control_gap)

        self.icon_label = IconLabel(16)
        self._layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignTop)

        self.text_label = QLabel("")
        self.text_label.setObjectName("statusBannerText")
        self.text_label.setWordWrap(True)
        self._layout.addWidget(self.text_label, 1)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(140)
        self._spinner_timer.timeout.connect(self._advance_spinner)

    def text(self) -> str:
        return self.text_label.text()

    def set_inline_mode(self) -> None:
        self.setProperty("density", "inline")
        self._layout.setContentsMargins(8, 5, 8, 5)
        self._layout.setAlignment(self.icon_label, Qt.AlignmentFlag.AlignVCenter)
        self._layout.setAlignment(self.text_label, Qt.AlignmentFlag.AlignVCenter)
        self.text_label.setWordWrap(True)
        self.style().unpolish(self)
        self.style().polish(self)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self.text_label.setText(text)

    def set_message(self, text: str, level: str = "info") -> None:
        self._stop_spinner()
        self._state = ""
        self.setProperty("state", "")
        normalized = level if level in self.VALID_LEVELS else "info"
        self.setProperty("level", normalized)
        self._base_text = text
        self.icon_label.set_semantic_icon(_level_icon(normalized))
        self.icon_label.setVisible(bool(text))
        self.setText(text)
        self.setVisible(bool(text))
        self.style().unpolish(self)
        self.style().polish(self)

    def refresh_icon(self) -> None:
        if self._state:
            self._render_state_text()
            return
        self.icon_label.set_semantic_icon(_level_icon(str(self.property("level") or "info")))

    def set_state(self, title: str, message: str = "", state: str = "idle", level: str | None = None) -> None:
        normalized_state = state if state in self.STATE_LEVELS else "idle"
        normalized_level = level if level in self.VALID_LEVELS else self.STATE_LEVELS[normalized_state]
        self._state = normalized_state
        self.setProperty("state", normalized_state)
        self.setProperty("level", normalized_level)
        self._base_text = f"{title}: {message}" if message else title
        self._spinner_index = 0
        self._render_state_text()
        self.icon_label.setVisible(bool(self._base_text))
        self.setVisible(bool(self._base_text))
        if normalized_state == "processing":
            self._spinner_timer.setInterval(140)
            self._spinner_timer.start()
        elif normalized_state == "live":
            self._spinner_timer.setInterval(700)
            self._spinner_timer.start()
        else:
            self._stop_spinner()
        self.style().unpolish(self)
        self.style().polish(self)

    def clear(self) -> None:
        self._stop_spinner()
        self._base_text = ""
        self._state = ""
        self.setProperty("state", "")
        self.setText("")
        self.icon_label.setPixmap(QPixmap())
        self.icon_label.setText("")
        self.icon_label.setVisible(False)
        self.setVisible(False)

    def _advance_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
        self._render_state_text()

    def _render_state_text(self) -> None:
        if self._state == "processing":
            marker = self.SPINNER_FRAMES[self._spinner_index]
            self.icon_label.setPixmap(QPixmap())
            self.icon_label.setText(marker)
        elif self._state == "live":
            marker = self.LIVE_FRAMES[self._spinner_index % len(self.LIVE_FRAMES)]
            self.icon_label.setPixmap(QPixmap())
            self.icon_label.setText(marker)
        else:
            self.icon_label.set_semantic_icon(self._state or "idle")
        self.setText(self._base_text)

    def _stop_spinner(self) -> None:
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()


def _level_icon(level: str) -> str:
    return {
        "info": "idle",
        "success": "ready",
        "warning": "blocked",
        "error": "error",
    }.get(level, "idle")


class PathPickerRow(QWidget):
    def __init__(
        self,
        button_text: str,
        callback: Callable[[], None] | None = None,
        *,
        label_text: str | None = None,
        read_only: bool = False,
    ) -> None:
        super().__init__()
        self.setObjectName("pathPickerRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().control_gap)

        if label_text:
            label = QLabel(label_text)
            label.setObjectName("pathPickerLabel")
            layout.addWidget(label)

        self.line_edit = QLineEdit()
        self.line_edit.setReadOnly(read_only)
        self.line_edit.setMinimumWidth(0)
        layout.addWidget(self.line_edit, 1)

        self.button = QPushButton(button_text)
        self.button.setObjectName("pathPickerButton")
        set_button_icon(self.button, "browse")
        if callback is not None:
            self.button.clicked.connect(callback)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, text: str) -> None:
        self.line_edit.setText(text)


class ActionButtonRow(QWidget):
    def __init__(self, buttons: Iterable[QPushButton] = (), *, align_right: bool = False) -> None:
        super().__init__()
        self.setObjectName("actionButtonRow")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(*zero_margins())
        self.layout.setSpacing(theme_spacing().control_gap)
        if align_right:
            self.layout.addStretch(1)
        for button in buttons:
            self.add_button(button)
        if not align_right:
            self.layout.addStretch(1)

    def add_button(self, button: QPushButton) -> None:
        self.layout.addWidget(button)


def set_button_role(button: QPushButton, role: str) -> QPushButton:
    button.setProperty("buttonRole", role)
    button.style().unpolish(button)
    button.style().polish(button)
    icon_name = button.property("appIconName")
    if icon_name:
        set_button_icon(button, str(icon_name), size=int(button.property("appIconSize") or 14))
    return button


def set_button_icon(button: QPushButton, icon_name: str, *, size: int = 14) -> QPushButton:
    color = active_theme().palette.text_on_accent if button.property("buttonRole") == "primary" else None
    icon = state_icon(icon_name, color=color)
    if not icon.isNull():
        button.setIcon(icon)
        button.setIconSize(QSize(size, size))
        button.setProperty("appIconName", icon_name)
        button.setProperty("appIconSize", size)
    return button


class ScrollablePane(QScrollArea):
    def __init__(self, widget: QWidget, *, object_name: str) -> None:
        super().__init__()
        self.setObjectName(object_name)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWidget(widget)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        dimensions = theme_dimensions()
        return QSize(dimensions.workflow_rail_size_hint_width, dimensions.workflow_rail_size_hint_height)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        dimensions = theme_dimensions()
        return QSize(dimensions.workflow_rail_min_width, dimensions.workflow_rail_min_height)


class MetricGrid(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("metricGrid")
        self._layout = QGridLayout(self)
        spacing = theme_spacing()
        self._layout.setContentsMargins(*zero_margins())
        self._layout.setHorizontalSpacing(spacing.control_gap * 2)
        self._layout.setVerticalSpacing(spacing.control_gap)
        self._value_widgets: dict[str, QLabel] = {}

    def set_metrics(self, metrics: Iterable[tuple[str, str | None]]) -> None:
        self.clear()
        metric_list = list(metrics)
        for index, (label_text, value_text) in enumerate(metric_list):
            row = index // 2
            column = 0 if index % 2 == 0 else 2

            label = QLabel(label_text)
            label.setObjectName("metricLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            value = QLabel("" if value_text is None else str(value_text))
            value.setObjectName("metricValue")
            value.setWordWrap(True)
            value.setMinimumHeight(theme_dimensions().metric_value_min_height)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            self._value_widgets[label_text] = value
            self._layout.addWidget(label, row, column)
            self._layout.addWidget(value, row, column + 1)

        self._layout.setColumnStretch(0, 0)
        self._layout.setColumnStretch(1, 1)
        self._layout.setColumnStretch(2, 0)
        self._layout.setColumnStretch(3, 1)

    def clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._value_widgets.clear()

    def value_text(self, label: str) -> str:
        widget = self._value_widgets.get(label)
        return "" if widget is None else widget.text()


class ResultBadge(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("resultBadge")
        self.setProperty("result", "neutral")
        self._text = "No result"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(theme_spacing().compact_control_gap + 2)

        self.icon_label = IconLabel(18)
        self.icon_label.set_result_icon("neutral")
        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.text_label = QLabel(self._text)
        self.text_label.setObjectName("resultBadgeText")
        layout.addWidget(self.text_label, 1, Qt.AlignmentFlag.AlignVCenter)

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._text = text
        self.text_label.setText(text)

    def set_result(self, result_name: str) -> None:
        normalized = result_name or "neutral"
        self.setProperty("result", normalized)
        self.setText(format_final_result(normalized) if result_name else "No result")
        self.icon_label.set_result_icon(normalized)
        self.style().unpolish(self)
        self.style().polish(self)

    def refresh_icon(self) -> None:
        self.icon_label.set_result_icon(str(self.property("result") or "neutral"))


class ResultSummaryPanel(SectionPanel):
    ESSENTIAL_METRIC_LABELS = {
        "Presence",
        "Anomaly decision",
        "Score",
        "Inspection time",
    }

    def __init__(self, *, min_height: int | None = None) -> None:
        super().__init__("Result Summary", compact=True)
        dimensions = theme_dimensions()
        self.setMinimumHeight(dimensions.result_summary_min_height if min_height is None else min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.operator_label = QLabel("Inspection Result")
        self.operator_label.setObjectName("resultDetailLabel")
        self.content_layout.addWidget(self.operator_label)

        self.result_badge = ResultBadge()
        self.content_layout.addWidget(self.result_badge)

        self.essential_label = QLabel("Essential Metrics")
        self.essential_label.setObjectName("resultDetailLabel")
        self.content_layout.addWidget(self.essential_label)

        self.essential_metric_grid = MetricGrid()
        self.metric_grid = self.essential_metric_grid
        self.content_layout.addWidget(self.essential_metric_grid)

        self.error_label = QLabel("Error Message")
        self.error_label.setObjectName("resultDetailLabel")
        self.error_text = QPlainTextEdit()
        self.error_text.setObjectName("resultDetailText")
        self.error_text.setReadOnly(True)
        self.error_text.setFixedHeight(dimensions.result_detail_error_height)
        self.error_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.content_layout.addWidget(self.error_label)
        self.content_layout.addWidget(self.error_text)
        self._set_error_visible(False)

        self.technical_button = set_button_role(QPushButton("Technical Details"), "secondary")
        self.technical_button.setObjectName("moreDetailsButton")
        self.technical_button.setCheckable(True)
        self.technical_button.toggled.connect(self._set_technical_visible)
        self.content_layout.addWidget(self.technical_button)

        self.technical_details_widget = QWidget()
        self.technical_details_widget.setObjectName("resultTechnicalDetails")
        technical_layout = QVBoxLayout(self.technical_details_widget)
        technical_layout.setContentsMargins(*zero_margins())
        technical_layout.setSpacing(theme_spacing().compact_control_gap)

        self.technical_metric_grid = MetricGrid()
        technical_layout.addWidget(self.technical_metric_grid)

        self.csv_label = QLabel("CSV Log Path")
        self.csv_label.setObjectName("resultDetailLabel")
        self.csv_text = QPlainTextEdit()
        self.csv_text.setObjectName("resultDetailText")
        self.csv_text.setReadOnly(True)
        self.csv_text.setFixedHeight(dimensions.result_detail_csv_height)
        self.csv_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        technical_layout.addWidget(self.csv_label)
        technical_layout.addWidget(self.csv_text)
        self.content_layout.addWidget(self.technical_details_widget)
        self._set_technical_available(False)

    def set_result(
        self,
        final_result: str,
        metrics: Iterable[tuple[str, str | None]],
        *,
        csv_log_path: str = "",
        error_message: str = "",
    ) -> None:
        result_name = final_result or "neutral"
        self.result_badge.set_result(result_name if final_result else "")

        essential_metrics, technical_metrics = self._split_metrics(metrics)
        self.essential_metric_grid.set_metrics(essential_metrics)
        self.technical_metric_grid.set_metrics(technical_metrics)
        self.csv_text.setPlainText(csv_log_path)
        self.error_text.setPlainText(error_message)
        self._set_error_visible(bool(error_message))
        self._set_technical_available(bool(technical_metrics) or bool(csv_log_path))

    def clear(self) -> None:
        self.result_badge.set_result("")
        self.essential_metric_grid.clear()
        self.technical_metric_grid.clear()
        self.csv_text.clear()
        self.error_text.clear()
        self._set_error_visible(False)
        self._set_technical_available(False)

    def _split_metrics(
        self,
        metrics: Iterable[tuple[str, str | None]],
    ) -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
        essential = []
        technical = []
        for label, value in metrics:
            target = essential if label in self.ESSENTIAL_METRIC_LABELS else technical
            target.append((label, value))
        return essential, technical

    def _set_error_visible(self, visible: bool) -> None:
        self.error_label.setVisible(visible)
        self.error_text.setVisible(visible)

    def _set_technical_available(self, available: bool) -> None:
        self.technical_button.setVisible(available)
        if not available:
            self.technical_button.setChecked(False)
        self._set_technical_visible(available and self.technical_button.isChecked())

    def _set_technical_visible(self, visible: bool) -> None:
        self.technical_details_widget.setVisible(visible)


def result_badge_text(result_name: str) -> str:
    return format_final_result(result_name)


class ImagePreviewWidget(QGraphicsView):
    MIN_ZOOM = 1.0
    MAX_ZOOM = 24.0
    ZOOM_FACTOR = 1.18

    def __init__(self, placeholder: str, *, min_height: int | None = None) -> None:
        super().__init__()
        self.setObjectName("imagePreviewWidget")
        self.setMinimumHeight(theme_dimensions().preview_min_height if min_height is None else min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._empty_icon_item: QGraphicsPixmapItem | None = None
        self._text_item: QGraphicsTextItem | None = None
        self._placeholder = placeholder
        self._empty_message = placeholder
        self._zoom = 1.0
        self._fit_mode = True
        self.clear_preview(placeholder)

    def set_image_path(self, path: str | Path | None, unavailable_message: str) -> None:
        if not path or not Path(path).exists():
            self.clear_preview(unavailable_message)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.clear_preview(f"Could not read image: {path}")
            return
        self._scene.clear()
        self._text_item = None
        self._empty_icon_item = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._zoom = 1.0
        self._fit_mode = True
        QTimer.singleShot(0, self.fit_current_image)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(320, self.minimumHeight())

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(64, 64)

    def clear_preview(self, message: str | None = None) -> None:
        self._empty_message = message or self._placeholder
        self._scene.clear()
        self._pixmap_item = None
        self._empty_icon_item = None
        self._zoom = 1.0
        self._fit_mode = True
        display_text = _empty_state_text(message or self._placeholder)
        empty_icon = icon_pixmap(state_icon(_empty_state_icon_name(display_text)), 24)
        self._text_item = self._scene.addText(display_text)
        self._text_item.setDefaultTextColor(Qt.GlobalColor.lightGray)
        text_rect = self._text_item.boundingRect()
        if not empty_icon.isNull():
            self._empty_icon_item = self._scene.addPixmap(empty_icon)
            self._empty_icon_item.setOffset(-empty_icon.width() / 2, -empty_icon.height() / 2)
            self._empty_icon_item.setPos(text_rect.width() / 2, -22)
            self._text_item.setPos(0, 0)
            rect = self._scene.itemsBoundingRect().adjusted(-8, -8, 8, 8)
        else:
            rect = text_rect
        self._scene.setSceneRect(rect)
        self.centerOn(rect.center())

    def refresh_icon(self) -> None:
        if self._pixmap_item is None:
            self.clear_preview(self._empty_message)

    def fit_current_image(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0
        self._fit_mode = True
        self.centerOn(self._pixmap_item)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._pixmap_item is None:
            super().wheelEvent(event)
            return
        zoom_in = event.angleDelta().y() > 0
        factor = self.ZOOM_FACTOR if zoom_in else 1.0 / self.ZOOM_FACTOR
        next_zoom = self._zoom * factor
        if next_zoom < self.MIN_ZOOM:
            if self._zoom != self.MIN_ZOOM:
                self.fit_current_image()
            return
        if next_zoom > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / self._zoom
            next_zoom = self.MAX_ZOOM
        self.scale(factor, factor)
        self._zoom = next_zoom
        self._fit_mode = self._zoom == self.MIN_ZOOM

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._fit_mode:
            QTimer.singleShot(0, self.fit_current_image)

    def refit_if_in_fit_mode(self) -> None:
        if self._fit_mode:
            QTimer.singleShot(0, self.fit_current_image)


class ResultImageTabs(QWidget):
    def __init__(
        self,
        *,
        annotated_placeholder: str = "No result images\nRun an inspection to generate annotated output.",
        heatmap_placeholder: str = "No heatmap\nRun an inspection to generate heatmap output.",
        mask_placeholder: str = "No presence mask\nRun an inspection to generate the presence mask.",
        min_height: int | None = None,
        tabs_min_height: int | None = None,
        preview_min_height: int | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("resultImageTabs")
        dimensions = theme_dimensions()
        self.setMinimumHeight(dimensions.result_image_tabs_min_height if min_height is None else min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().control_gap)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("resultImageTabWidget")
        self.tabs.setMinimumHeight(dimensions.result_tabs_min_height if tabs_min_height is None else tabs_min_height)
        self.fit_image_button = set_button_role(QPushButton("Fit Image"), "secondary")
        self.fit_image_button.setObjectName("fitImageButton")
        set_button_icon(self.fit_image_button, "fit")
        self.fit_image_button.clicked.connect(self.fit_active_preview)
        self.tabs.setCornerWidget(self.fit_image_button, Qt.Corner.TopRightCorner)

        self.annotated_preview = ImagePreviewWidget(annotated_placeholder, min_height=preview_min_height)
        self.heatmap_preview = ImagePreviewWidget(heatmap_placeholder, min_height=preview_min_height)
        self.mask_preview = ImagePreviewWidget(mask_placeholder, min_height=preview_min_height)
        self.tabs.addTab(self.annotated_preview, "Annotated")
        self.tabs.addTab(self.heatmap_preview, "Heatmap")
        self.tabs.addTab(self.mask_preview, "Presence Mask")
        self.tabs.currentChanged.connect(self.refit_active_preview_if_needed)
        layout.addWidget(self.tabs, 1)

        self._annotated_placeholder = annotated_placeholder
        self._heatmap_placeholder = heatmap_placeholder
        self._mask_placeholder = mask_placeholder

    def set_artifacts(
        self,
        *,
        annotated_path: str | Path | None = None,
        heatmap_path: str | Path | None = None,
        presence_mask_path: str | Path | None = None,
    ) -> None:
        self.annotated_preview.set_image_path(annotated_path, "No annotated image generated for this result.")
        self.heatmap_preview.set_image_path(heatmap_path, "No heatmap generated for this result.")
        self.mask_preview.set_image_path(presence_mask_path, "No presence mask generated for this result.")

    def clear(self) -> None:
        self.annotated_preview.clear_preview(self._annotated_placeholder)
        self.heatmap_preview.clear_preview(self._heatmap_placeholder)
        self.mask_preview.clear_preview(self._mask_placeholder)

    def fit_active_preview(self) -> None:
        widget = self.tabs.currentWidget()
        if isinstance(widget, ImagePreviewWidget):
            widget.fit_current_image()

    def refit_active_preview_if_needed(self) -> None:
        widget = self.tabs.currentWidget()
        if isinstance(widget, ImagePreviewWidget):
            widget.refit_if_in_fit_mode()


def _empty_state_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "No content"
    return stripped


def _empty_state_icon_name(text: str) -> str:
    lowered = text.lower()
    if lowered.startswith("could not"):
        return "error"
    if "camera" in lowered:
        return "camera"
    if "log" in lowered or "record" in lowered:
        return "logs"
    if "zone" in lowered or "polygon" in lowered:
        return "zone"
    if "image" in lowered or "heatmap" in lowered or "mask" in lowered:
        return "image"
    return "empty"
