from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

TOOLBAR_HEIGHT = 88
MAX_CANVAS_SIZE = (1920, 1080)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class ToolbarButton:
    action: str
    label: str
    rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class DisplayLayout:
    canvas_width: int
    canvas_height: int
    source_width: int
    source_height: int
    image_rect: tuple[int, int, int, int]
    toolbar_rect: tuple[int, int, int, int]
    scale: float


def initial_canvas_size(source_shape: tuple[int, ...], max_size: tuple[int, int] = MAX_CANVAS_SIZE) -> tuple[int, int]:
    source_height, source_width = source_shape[:2]
    max_width, max_height = max_size
    available_height = max(1, max_height - TOOLBAR_HEIGHT)
    scale = min(1.0, max_width / source_width, available_height / source_height)
    display_width = max(1, int(round(source_width * scale)))
    display_height = max(1, int(round(source_height * scale)))
    return display_width, min(max_height, display_height + TOOLBAR_HEIGHT)


def toolbar_height_for_canvas(canvas_height: int) -> int:
    if canvas_height >= 260:
        return TOOLBAR_HEIGHT
    if canvas_height >= 170:
        return 64
    return max(44, canvas_height // 3)


def compute_display_layout(source_shape: tuple[int, ...], canvas_size: tuple[int, int]) -> DisplayLayout:
    source_height, source_width = source_shape[:2]
    canvas_width = max(80, int(canvas_size[0]))
    canvas_height = max(80, int(canvas_size[1]))
    toolbar_height = min(toolbar_height_for_canvas(canvas_height), max(40, canvas_height - 24))
    image_area_height = max(1, canvas_height - toolbar_height)
    scale = min(canvas_width / source_width, image_area_height / source_height)
    display_width = max(1, int(round(source_width * scale)))
    display_height = max(1, int(round(source_height * scale)))
    image_x = max(0, (canvas_width - display_width) // 2)
    image_y = max(0, (image_area_height - display_height) // 2)
    return DisplayLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        source_width=source_width,
        source_height=source_height,
        image_rect=(image_x, image_y, display_width, display_height),
        toolbar_rect=(0, image_area_height, canvas_width, toolbar_height),
        scale=scale,
    )


def window_canvas_size(window_name: str, fallback: tuple[int, int], max_size: tuple[int, int] = MAX_CANVAS_SIZE) -> tuple[int, int]:
    try:
        _, _, width, height = cv2.getWindowImageRect(window_name)
        if width > 0 and height > 0:
            capped_width = min(width, max_size[0])
            capped_height = min(height, max_size[1])
            if capped_width != width or capped_height != height:
                cv2.resizeWindow(window_name, capped_width, capped_height)
            width = capped_width
            height = capped_height
            return width, height
    except Exception:
        pass
    return min(fallback[0], max_size[0]), min(fallback[1], max_size[1])


def layout_toolbar_buttons(
    width: int,
    actions: Sequence[tuple[str, str]],
    toolbar_height: int = TOOLBAR_HEIGHT,
    toolbar_y: int = 0,
) -> list[ToolbarButton]:
    if width <= 0:
        raise ValueError("Toolbar width must be positive.")
    margin = 8
    gap = 8
    font_scale = 0.46 if width >= 700 else 0.40
    pad_x = 10
    button_height = 34
    specs = list(actions)

    def total_width(scale: float, padding: int, spacing: int) -> int:
        widths = [cv2.getTextSize(label, FONT, scale, 1)[0][0] + padding * 2 for _, label in specs]
        return sum(widths) + spacing * max(0, len(widths) - 1) + margin * 2

    if total_width(font_scale, pad_x, gap) > width:
        font_scale = 0.36
        pad_x = 6
        gap = 5

    text_widths = [cv2.getTextSize(label, FONT, font_scale, 1)[0][0] + pad_x * 2 for _, label in specs]
    available = width - margin * 2 - gap * max(0, len(specs) - 1)
    if sum(text_widths) > available and specs:
        button_widths = [max(28, available // len(specs)) for _ in specs]
    else:
        button_widths = text_widths

    buttons: list[ToolbarButton] = []
    x = margin
    y1 = toolbar_y + 10
    y2 = toolbar_y + min(10 + button_height, toolbar_height - 28)
    for (action, label), button_width in zip(specs, button_widths):
        x2 = min(width - margin, x + button_width)
        buttons.append(ToolbarButton(action=action, label=label, rect=(x, y1, x2, y2)))
        x = x2 + gap
    return buttons


def draw_toolbar_on_canvas(
    canvas: np.ndarray,
    buttons: Sequence[ToolbarButton],
    toolbar_rect: tuple[int, int, int, int],
    status: str = "",
) -> None:
    x, y, width, height = toolbar_rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (42, 42, 42), thickness=-1)
    cv2.line(canvas, (x, y), (x + width, y), (95, 95, 95), 1, cv2.LINE_AA)
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (72, 72, 72), thickness=-1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), thickness=1)
        text_size, _ = cv2.getTextSize(button.label, FONT, 0.42, 1)
        text_x = x1 + max(4, ((x2 - x1) - text_size[0]) // 2)
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2
        cv2.putText(canvas, button.label, (text_x, text_y), FONT, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    if status:
        cv2.putText(canvas, (status[:160]), (x + 10, y + height - 14), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


def draw_toolbar(width: int, buttons: Sequence[ToolbarButton], status: str = "", toolbar_height: int = TOOLBAR_HEIGHT) -> np.ndarray:
    toolbar = np.full((toolbar_height, width, 3), (42, 42, 42), dtype=np.uint8)
    draw_toolbar_on_canvas(toolbar, buttons, (0, 0, width, toolbar_height), status=status)
    return toolbar


def compose_with_toolbar(image: np.ndarray, buttons: Sequence[ToolbarButton], status: str = "") -> np.ndarray:
    toolbar = draw_toolbar(image.shape[1], buttons, status=status)
    return np.vstack([image, toolbar])


def render_interactive_canvas(
    source_image: np.ndarray,
    actions: Sequence[tuple[str, str]],
    status: str,
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, DisplayLayout, list[ToolbarButton]]:
    layout = compute_display_layout(source_image.shape, canvas_size)
    canvas = np.full((layout.canvas_height, layout.canvas_width, 3), (28, 28, 28), dtype=np.uint8)
    image_x, image_y, image_width, image_height = layout.image_rect
    resized = cv2.resize(source_image, (image_width, image_height), interpolation=cv2.INTER_AREA if layout.scale < 1 else cv2.INTER_LINEAR)
    canvas[image_y : image_y + image_height, image_x : image_x + image_width] = resized
    toolbar_x, toolbar_y, toolbar_width, toolbar_height = layout.toolbar_rect
    buttons = layout_toolbar_buttons(toolbar_width, actions, toolbar_height=toolbar_height, toolbar_y=toolbar_y)
    draw_toolbar_on_canvas(canvas, buttons, layout.toolbar_rect, status=status)
    return canvas, layout, buttons


def toolbar_action_at(buttons: Sequence[ToolbarButton], x: int, y: int, image_height: int | None = None) -> str | None:
    if image_height is not None:
        if y < image_height:
            return None
        if all(button.rect[1] < image_height for button in buttons):
            y = y - image_height
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button.action
    return None


def display_to_source(layout: DisplayLayout, x: int, y: int) -> tuple[int, int] | None:
    image_x, image_y, image_width, image_height = layout.image_rect
    if not (image_x <= x < image_x + image_width and image_y <= y < image_y + image_height):
        return None
    source_x = int((x - image_x) / layout.scale)
    source_y = int((y - image_y) / layout.scale)
    if not (0 <= source_x < layout.source_width and 0 <= source_y < layout.source_height):
        return None
    return source_x, source_y
