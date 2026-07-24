from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anomaly_inspection.cli_support.toolbar_ui import (
    DisplayLayout,
    ToolbarButton,
    initial_canvas_size,
    render_interactive_canvas,
    toolbar_action_at,
    window_canvas_size,
)

LIVE_ACTIONS = (("capture", "Capture Background"), ("quit", "Quit"))
CAPTURED_ACTIONS = (("save", "Use as Reference"), ("retake", "Retake"), ("quit", "Quit"))


def save_reference_image(frame: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise ValueError(f"Could not save reference image to {output_path}")
    return output_path


class ReferenceCaptureUI:
    def __init__(self, output_path: str | Path, camera_index: int = 0, width: int | None = None, height: int | None = None):
        self.output_path = Path(output_path)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.window_name = "Capture Empty Reference"
        self.state = "live"
        self.status = "Point camera at empty station, then capture."
        self.current_frame: np.ndarray | None = None
        self.captured_frame: np.ndarray | None = None
        self.buttons: list[ToolbarButton] = []
        self.layout: DisplayLayout | None = None
        self.canvas_size = (640, 480 + 88)
        self.running = True

    def run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise ValueError(f"Could not open camera index {self.camera_index}.")
        try:
            if self.width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Could not read initial camera frame.")
            self.current_frame = frame
            self._report_resolution(frame)
            self.canvas_size = initial_canvas_size(frame.shape)
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.canvas_size[0], self.canvas_size[1])
            cv2.setMouseCallback(self.window_name, self._on_mouse)

            while self.running:
                if self.state == "live":
                    ok, frame = capture.read()
                    if not ok:
                        self.status = "Could not read camera frame."
                        key = cv2.waitKey(30) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            break
                        continue
                    self.current_frame = frame
                    frame_to_render = frame
                else:
                    frame_to_render = self.captured_frame
                self.canvas_size = window_canvas_size(self.window_name, self.canvas_size)
                canvas = self._render(frame_to_render, self.canvas_size)
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
        finally:
            capture.release()
            cv2.destroyWindow(self.window_name)

    def _report_resolution(self, frame: np.ndarray) -> None:
        actual_width, actual_height = frame.shape[1], frame.shape[0]
        if self.width and self.height and (actual_width != self.width or actual_height != self.height):
            print(
                f"Requested camera resolution {self.width}x{self.height}; "
                f"actual frame resolution is {actual_width}x{actual_height}."
            )

    def _render(self, frame: np.ndarray | None, canvas_size: tuple[int, int]) -> np.ndarray:
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        actions = LIVE_ACTIONS if self.state == "live" else CAPTURED_ACTIONS
        canvas, self.layout, self.buttons = render_interactive_canvas(frame, actions, self.status, canvas_size)
        return canvas

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        frame = self.current_frame if self.state == "live" else self.captured_frame
        if frame is None:
            return
        action = toolbar_action_at(self.buttons, x, y)
        if action:
            self._handle_action(action)

    def _handle_action(self, action: str) -> None:
        if action == "capture" and self.current_frame is not None:
            self.captured_frame = self.current_frame.copy()
            self.state = "captured"
            self.status = "Review captured frame. Save, retake, or quit."
        elif action == "retake":
            self.captured_frame = None
            self.state = "live"
            self.status = "Retaking. Point camera at empty station, then capture."
        elif action == "save" and self.captured_frame is not None:
            output_path = save_reference_image(self.captured_frame, self.output_path)
            print(f"Saved reference image to {output_path}")
            print(f"Reference image resolution: {self.captured_frame.shape[1]}x{self.captured_frame.shape[0]}")
            self.running = False
        elif action == "quit":
            self.running = False


def run_reference_capture(output_path: str | Path, camera_index: int = 0, width: int | None = None, height: int | None = None) -> None:
    ReferenceCaptureUI(output_path=output_path, camera_index=camera_index, width=width, height=height).run()
