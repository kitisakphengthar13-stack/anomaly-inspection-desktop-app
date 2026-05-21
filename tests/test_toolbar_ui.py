import numpy as np

from inspection.toolbar_ui import (
    MAX_CANVAS_SIZE,
    compute_display_layout,
    compose_with_toolbar,
    display_to_source,
    initial_canvas_size,
    layout_toolbar_buttons,
    render_interactive_canvas,
    toolbar_action_at,
)


def test_toolbar_hit_testing_separates_image_and_toolbar_regions():
    buttons = layout_toolbar_buttons(640, (("save", "Save Zones"), ("quit", "Quit")))

    assert toolbar_action_at(buttons, 20, 20, image_height=480) is None

    x1, y1, x2, y2 = buttons[0].rect
    assert toolbar_action_at(buttons, x1 + 2, 480 + y1 + 2, image_height=480) == "save"
    assert toolbar_action_at(buttons, 639, 480 + y2 + 20, image_height=480) is None


def test_compose_with_toolbar_preserves_full_image_area():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    buttons = layout_toolbar_buttons(640, (("capture", "Capture Background"), ("quit", "Quit")))

    canvas = compose_with_toolbar(image, buttons, "status")

    assert canvas.shape[0] > image.shape[0]
    assert canvas.shape[1] == image.shape[1]
    assert np.array_equal(canvas[: image.shape[0], : image.shape[1]], image)


def test_initial_canvas_size_uses_native_size_when_within_limit():
    assert initial_canvas_size((640, 640, 3)) == (640, 728)
    assert initial_canvas_size((850, 1700, 3)) == (1700, 938)


def test_initial_canvas_size_scales_down_to_maximum():
    width, height = initial_canvas_size((2200, 4000, 3))

    assert width <= MAX_CANVAS_SIZE[0]
    assert height <= MAX_CANVAS_SIZE[1]


def test_display_layout_scales_and_maps_points_to_source_coordinates():
    layout = compute_display_layout((1000, 2000, 3), (1000, 600))

    assert layout.scale < 1
    x, y, width, height = layout.image_rect
    assert display_to_source(layout, x, y) == (0, 0)
    assert display_to_source(layout, x + width - 1, y + height - 1) == (1998, 998)
    assert display_to_source(layout, 0, layout.toolbar_rect[1] + 2) is None


def test_rendered_toolbar_hit_testing_uses_current_canvas_coordinates():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    _, _, buttons_small = render_interactive_canvas(image, (("save", "Save Zones"), ("quit", "Quit")), "status", (320, 240))
    _, _, buttons_large = render_interactive_canvas(image, (("save", "Save Zones"), ("quit", "Quit")), "status", (1280, 900))

    for buttons in (buttons_small, buttons_large):
        x1, y1, _, _ = buttons[0].rect
        assert toolbar_action_at(buttons, x1 + 1, y1 + 1) == "save"
