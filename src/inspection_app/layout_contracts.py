from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from inspection_app.theme import theme_dimensions, theme_spacing, zero_margins
from inspection_app.ui_components import ActionButtonRow, ScrollablePane


class ElidedValueLabel(QLabel):
    """Compact label for long values that should not drive parent geometry."""

    def __init__(
        self,
        text: str = "-",
        *,
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle,
    ) -> None:
        super().__init__()
        self.setObjectName("elidedValueLabel")
        self._full_text = text or "-"
        self._mode = mode
        self.setWordWrap(False)
        self.setToolTip(self._full_text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._update_text()

    def set_full_text(self, text: str) -> None:
        self._full_text = text or "-"
        self.setToolTip(self._full_text)
        self._update_text()

    def full_text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_text()

    def _update_text(self) -> None:
        metrics = QFontMetrics(self.font())
        self.setText(metrics.elidedText(self._full_text, self._mode, max(10, self.width())))


class ActionSection(QWidget):
    """Fixed-height section for workflow-critical action rows."""

    def __init__(self, *, rows: int = 1) -> None:
        super().__init__()
        self.setObjectName("actionSection")
        self._rows = max(1, rows)
        self._row_widgets: list[ActionButtonRow] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*zero_margins())
        self._layout.setSpacing(theme_spacing().control_gap)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._apply_height_contract()

    def add_row(self, buttons: Iterable[QPushButton], *, align_right: bool = False) -> ActionButtonRow:
        row = ActionButtonRow(buttons, align_right=align_right)
        for button in buttons:
            button.setMinimumHeight(theme_dimensions().button_min_height)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._layout.addWidget(row)
        self._row_widgets.append(row)
        self._apply_height_contract()
        return row

    def _apply_height_contract(self) -> None:
        dimensions = theme_dimensions()
        spacing = theme_spacing().control_gap
        margins = self._layout.contentsMargins()
        reserved_row_height = dimensions.button_min_height
        if self._row_widgets:
            reserved_row_height = max(reserved_row_height, *(row.sizeHint().height() for row in self._row_widgets))
        row_count = max(self._rows, len(self._row_widgets))
        height = (
            row_count * reserved_row_height
            + max(0, row_count - 1) * spacing
            + margins.top()
            + margins.bottom()
        )
        self.setMinimumHeight(height)


class ControlRail(QWidget):
    """Bounded side panel with fixed critical sections and an optional scroll body."""

    def __init__(
        self,
        *,
        object_name: str,
        min_width: int | None = None,
        max_width: int | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName(object_name)
        dimensions = theme_dimensions()
        self.setMinimumWidth(min_width or dimensions.workflow_rail_min_width)
        self.setMaximumWidth(max_width or dimensions.workflow_rail_size_hint_width)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*zero_margins())
        self._layout.setSpacing(theme_spacing().left_pane_gap)
        self.scroll_body: ScrollablePane | None = None

    def add_fixed(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def set_scroll_body(self, widget: QWidget, *, object_name: str) -> ScrollablePane:
        scroll = ScrollablePane(widget, object_name=object_name)
        self.scroll_body = scroll
        self._layout.addWidget(scroll, 1)
        return scroll


class WorkbenchLayout(QSplitter):
    """Horizontal workbench with a bounded rail and expanding main workspace."""

    def __init__(self, *, object_name: str, rail_position: Literal["left", "right"] = "left") -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setObjectName(object_name)
        self.setChildrenCollapsible(False)
        self._rail_position = rail_position
        self._rail: ControlRail | None = None
        self._main_workspace: QWidget | None = None

    def set_control_rail(self, rail: ControlRail) -> None:
        self._rail = rail
        self._rebuild()

    def set_main_workspace(self, widget: QWidget) -> None:
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._main_workspace = widget
        self._rebuild()

    def _rebuild(self) -> None:
        if self._rail is None or self._main_workspace is None:
            return
        if self.count():
            return
        if self._rail_position == "left":
            self.addWidget(self._rail)
            self.addWidget(self._main_workspace)
        else:
            self.addWidget(self._main_workspace)
            self.addWidget(self._rail)
        self.setStretchFactor(0, 1 if self._rail_position == "left" else 4)
        self.setStretchFactor(1, 4 if self._rail_position == "left" else 1)
        self.setSizes([self._rail.width(), self._main_workspace.width()])
