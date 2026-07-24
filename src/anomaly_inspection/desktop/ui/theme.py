from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class SemanticColors:
    foreground: str
    background: str
    border: str


@dataclass(frozen=True)
class ZoneOverlayPalette:
    finalized: tuple[int, int, int]
    finalized_fill: tuple[int, int, int, int]
    finalized_label: tuple[int, int, int]
    current: tuple[int, int, int]


@dataclass(frozen=True)
class ThemePalette:
    app_background: str
    shell_background: str
    surface: str
    surface_elevated: str
    surface_hover: str
    surface_inset: str
    surface_disabled: str
    canvas_surface: str
    table_alternate: str
    table_header: str
    table_text: str
    border: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_muted: str
    text_on_accent: str
    text_on_result: str
    accent: str
    accent_muted: str
    accent_button: str
    accent_button_hover: str
    accent_button_hover_border: str
    danger_button_foreground: str
    danger_button_background: str
    success: str
    warning: str
    danger: str
    selection: str
    selection_text: str
    status: Mapping[str, SemanticColors] = field(default_factory=dict)
    result: Mapping[str, SemanticColors] = field(default_factory=dict)
    zone_overlay: ZoneOverlayPalette = field(
        default_factory=lambda: ZoneOverlayPalette(
            finalized=(255, 205, 0),
            finalized_fill=(255, 205, 0, 45),
            finalized_label=(255, 245, 180),
            current=(20, 220, 120),
        )
    )


@dataclass(frozen=True)
class ThemeSpacing:
    page_margin_left: int
    page_margin_top: int
    page_margin_right: int
    page_margin_bottom: int
    setup_page_gap: int
    operations_page_gap: int
    section_gap: int
    panel_padding: int
    control_gap: int
    compact_panel_padding: int
    compact_control_gap: int
    left_pane_gap: int
    compact_grid_horizontal_gap: int
    compact_grid_vertical_gap: int
    logs_review_gap: int
    logs_filter_gap: int
    group_margin_left: int
    group_margin_top: int
    group_margin_right: int
    group_margin_bottom: int
    group_content_margin_left: int
    group_content_margin_top: int
    group_content_margin_right: int
    group_content_margin_bottom: int
    shell_margin_horizontal: int
    shell_margin_vertical: int
    shell_gap: int
    nav_cluster_gap: int


@dataclass(frozen=True)
class ThemeTypography:
    page_title_px: int
    page_subtitle_px: int
    section_title_px: int
    section_title_compact_px: int
    section_subtitle_px: int
    section_subtitle_compact_px: int
    status_px: int
    metric_label_px: int
    metric_value_px: int
    result_badge_px: int
    shell_title_px: int
    shell_subtitle_px: int
    nav_group_px: int


@dataclass(frozen=True)
class ThemeDimensions:
    window_initial_width: int
    window_initial_height: int
    window_min_width: int
    window_min_height: int
    radius: int
    input_radius: int
    scrollbar_size: int
    scrollbar_radius: int
    nav_button_min_height: int
    button_min_height: int
    metric_value_min_height: int
    camera_preview_base_min_height: int
    reference_preview_min_height: int
    zone_canvas_min_height: int
    inspect_summary_min_height: int
    inspect_result_panel_min_height: int
    inspect_result_tabs_min_height: int
    inspect_result_image_tabs_min_height: int
    inspect_result_preview_min_height: int
    inspect_splitter_left_width: int
    inspect_splitter_right_width: int
    inspect_splitter_left_stretch: int
    inspect_splitter_right_stretch: int
    camera_preview_idle_min_height: int
    camera_workspace_min_height: int
    camera_visual_preview_min_height: int
    camera_result_tabs_min_height: int
    camera_result_image_tabs_min_height: int
    camera_result_preview_min_height: int
    camera_splitter_left_width: int
    camera_splitter_right_width: int
    camera_splitter_left_stretch: int
    camera_splitter_right_stretch: int
    workflow_rail_size_hint_width: int
    workflow_rail_size_hint_height: int
    workflow_rail_min_width: int
    workflow_rail_min_height: int
    logs_splitter_records_width: int
    logs_splitter_review_width: int
    logs_splitter_records_stretch: int
    logs_splitter_review_stretch: int
    logs_records_min_height: int
    logs_artifact_min_height: int
    logs_selected_record_context_max_height: int
    logs_details_scroll_max_height: int
    logs_table_row_height: int
    logs_table_header_min_section_width: int
    logs_timestamp_column_width: int
    logs_mode_column_width: int
    logs_result_column_width: int
    logs_presence_column_width: int
    logs_score_column_width: int
    logs_total_time_column_width: int
    logs_source_text_height: int
    logs_image_path_text_height: int
    logs_error_text_height: int
    preview_min_height: int
    result_tabs_min_height: int
    result_image_tabs_min_height: int
    result_summary_min_height: int
    result_detail_csv_height: int
    result_detail_error_height: int


@dataclass(frozen=True)
class InspectionTheme:
    name: str
    palette: ThemePalette
    spacing: ThemeSpacing
    typography: ThemeTypography
    dimensions: ThemeDimensions


FACTORY_DARK = InspectionTheme(
    name="factory_dark",
    palette=ThemePalette(
        app_background="#111820",
        shell_background="#0f1720",
        surface="#1b232d",
        surface_elevated="#202a35",
        surface_hover="#273241",
        surface_inset="#18212b",
        surface_disabled="#151d26",
        canvas_surface="#20242a",
        table_alternate="#252b33",
        table_header="#2b333d",
        table_text="#d8dee6",
        border="#354150",
        border_strong="#4a5563",
        text_primary="#e5e7eb",
        text_secondary="#aab4c0",
        text_muted="#7f8a99",
        text_on_accent="#e0f2fe",
        text_on_result="#f8fafc",
        accent="#38bdf8",
        accent_muted="#15384a",
        accent_button="#155e75",
        accent_button_hover="#166b84",
        accent_button_hover_border="#7dd3fc",
        danger_button_foreground="#fee2e2",
        danger_button_background="#5b1c20",
        success="#22c55e",
        warning="#f59e0b",
        danger="#ef4444",
        selection="#334155",
        selection_text="#f8fafc",
        status=MappingProxyType(
            {
                "info": SemanticColors("#bae6fd", "#0f2d3b", "#1f5f78"),
                "success": SemanticColors("#bbf7d0", "#123322", "#236642"),
                "warning": SemanticColors("#fde68a", "#3a2a0d", "#8a6415"),
                "error": SemanticColors("#fecaca", "#3b1518", "#7f1d1d"),
            }
        ),
        result=MappingProxyType(
            {
                "OK": SemanticColors("#f8fafc", "#166534", "#22c55e"),
                "NG": SemanticColors("#f8fafc", "#7f1d1d", "#ef4444"),
                "NO_PART": SemanticColors("#f8fafc", "#78350f", "#f59e0b"),
                "ERROR": SemanticColors("#f8fafc", "#7f1d1d", "#ef4444"),
                "neutral": SemanticColors("#f8fafc", "#334155", "#4a5563"),
            }
        ),
    ),
    spacing=ThemeSpacing(
        page_margin_left=18,
        page_margin_top=10,
        page_margin_right=24,
        page_margin_bottom=18,
        setup_page_gap=12,
        operations_page_gap=16,
        section_gap=16,
        panel_padding=14,
        control_gap=8,
        compact_panel_padding=10,
        compact_control_gap=5,
        left_pane_gap=10,
        compact_grid_horizontal_gap=8,
        compact_grid_vertical_gap=6,
        logs_review_gap=8,
        logs_filter_gap=6,
        group_margin_left=8,
        group_margin_top=10,
        group_margin_right=8,
        group_margin_bottom=7,
        group_content_margin_left=8,
        group_content_margin_top=12,
        group_content_margin_right=8,
        group_content_margin_bottom=8,
        shell_margin_horizontal=18,
        shell_margin_vertical=6,
        shell_gap=16,
        nav_cluster_gap=8,
    ),
    typography=ThemeTypography(
        page_title_px=24,
        page_subtitle_px=13,
        section_title_px=15,
        section_title_compact_px=14,
        section_subtitle_px=12,
        section_subtitle_compact_px=11,
        status_px=13,
        metric_label_px=12,
        metric_value_px=13,
        result_badge_px=18,
        shell_title_px=18,
        shell_subtitle_px=12,
        nav_group_px=11,
    ),
    dimensions=ThemeDimensions(
        window_initial_width=1280,
        window_initial_height=800,
        window_min_width=960,
        window_min_height=640,
        radius=6,
        input_radius=4,
        scrollbar_size=12,
        scrollbar_radius=5,
        nav_button_min_height=28,
        button_min_height=28,
        metric_value_min_height=20,
        camera_preview_base_min_height=320,
        reference_preview_min_height=480,
        zone_canvas_min_height=360,
        inspect_summary_min_height=150,
        inspect_result_panel_min_height=460,
        inspect_result_tabs_min_height=390,
        inspect_result_image_tabs_min_height=420,
        inspect_result_preview_min_height=340,
        inspect_splitter_left_width=560,
        inspect_splitter_right_width=860,
        inspect_splitter_left_stretch=2,
        inspect_splitter_right_stretch=4,
        camera_preview_idle_min_height=420,
        camera_workspace_min_height=540,
        camera_visual_preview_min_height=320,
        camera_result_tabs_min_height=470,
        camera_result_image_tabs_min_height=500,
        camera_result_preview_min_height=420,
        camera_splitter_left_width=950,
        camera_splitter_right_width=1000,
        camera_splitter_left_stretch=3,
        camera_splitter_right_stretch=4,
        workflow_rail_size_hint_width=560,
        workflow_rail_size_hint_height=420,
        workflow_rail_min_width=430,
        workflow_rail_min_height=180,
        logs_splitter_records_width=920,
        logs_splitter_review_width=650,
        logs_splitter_records_stretch=6,
        logs_splitter_review_stretch=4,
        logs_records_min_height=520,
        logs_artifact_min_height=560,
        logs_selected_record_context_max_height=285,
        logs_details_scroll_max_height=190,
        logs_table_row_height=30,
        logs_table_header_min_section_width=72,
        logs_timestamp_column_width=170,
        logs_mode_column_width=90,
        logs_result_column_width=90,
        logs_presence_column_width=130,
        logs_score_column_width=125,
        logs_total_time_column_width=125,
        logs_source_text_height=34,
        logs_image_path_text_height=38,
        logs_error_text_height=48,
        preview_min_height=560,
        result_tabs_min_height=620,
        result_image_tabs_min_height=680,
        result_summary_min_height=220,
        result_detail_csv_height=44,
        result_detail_error_height=58,
    ),
)

INDUSTRIAL_LIGHT_DEVCPP = InspectionTheme(
    name="industrial_light_devcpp",
    palette=ThemePalette(
        app_background="#f3f3f3",
        shell_background="#e6e6e6",
        surface="#f8f8f8",
        surface_elevated="#ffffff",
        surface_hover="#eceff3",
        surface_inset="#f1f1f1",
        surface_disabled="#e8e8e8",
        canvas_surface="#ffffff",
        table_alternate="#f2f5f8",
        table_header="#e1e5ea",
        table_text="#1f2933",
        border="#c4c9cf",
        border_strong="#9aa3ad",
        text_primary="#17212b",
        text_secondary="#3f4b57",
        text_muted="#687480",
        text_on_accent="#ffffff",
        text_on_result="#ffffff",
        accent="#2b6cb0",
        accent_muted="#dceaf8",
        accent_button="#2f6fa8",
        accent_button_hover="#255f93",
        accent_button_hover_border="#174d7f",
        danger_button_foreground="#ffffff",
        danger_button_background="#b42318",
        success="#14803f",
        warning="#b86b00",
        danger="#b42318",
        selection="#2b6cb0",
        selection_text="#ffffff",
        status=MappingProxyType(
            {
                "info": SemanticColors("#17324d", "#e3eef8", "#7aa7d2"),
                "success": SemanticColors("#0f3d23", "#dff3e7", "#61a978"),
                "warning": SemanticColors("#4a3200", "#fff1cf", "#d79b2e"),
                "error": SemanticColors("#5b1111", "#f8dddd", "#c15b5b"),
            }
        ),
        result=MappingProxyType(
            {
                "OK": SemanticColors("#ffffff", "#16803f", "#0f5e2d"),
                "NG": SemanticColors("#ffffff", "#c62828", "#8f1d1d"),
                "NO_PART": SemanticColors("#ffffff", "#b86b00", "#8a5100"),
                "ERROR": SemanticColors("#ffffff", "#8f1d1d", "#5d1010"),
                "neutral": SemanticColors("#17212b", "#d8dde3", "#9aa3ad"),
            }
        ),
        zone_overlay=ZoneOverlayPalette(
            finalized=(230, 137, 0),
            finalized_fill=(230, 137, 0, 38),
            finalized_label=(115, 65, 0),
            current=(0, 128, 92),
        ),
    ),
    spacing=FACTORY_DARK.spacing,
    typography=FACTORY_DARK.typography,
    dimensions=replace(FACTORY_DARK.dimensions, radius=3, input_radius=2, scrollbar_radius=3),
)

HIGH_CONTRAST_FACTORY_LIGHT = InspectionTheme(
    name="high_contrast_factory_light",
    palette=ThemePalette(
        app_background="#f7f8fa",
        shell_background="#dfe4ea",
        surface="#ffffff",
        surface_elevated="#ffffff",
        surface_hover="#e5ebf2",
        surface_inset="#eef2f6",
        surface_disabled="#d9dee5",
        canvas_surface="#ffffff",
        table_alternate="#edf2f7",
        table_header="#cfd8e3",
        table_text="#111827",
        border="#9aa6b2",
        border_strong="#5f6f80",
        text_primary="#111827",
        text_secondary="#263442",
        text_muted="#4c5d6e",
        text_on_accent="#ffffff",
        text_on_result="#ffffff",
        accent="#005a9e",
        accent_muted="#cfe5f7",
        accent_button="#005a9e",
        accent_button_hover="#004a82",
        accent_button_hover_border="#003a66",
        danger_button_foreground="#ffffff",
        danger_button_background="#9f1d1d",
        success="#007a3d",
        warning="#a35a00",
        danger="#b00020",
        selection="#005a9e",
        selection_text="#ffffff",
        status=MappingProxyType(
            {
                "info": SemanticColors("#092f57", "#d6ebff", "#2f75b5"),
                "success": SemanticColors("#063b1f", "#d5f1df", "#1b7f42"),
                "warning": SemanticColors("#412900", "#ffe6ad", "#b36a00"),
                "error": SemanticColors("#4a0909", "#ffd6d6", "#b00020"),
            }
        ),
        result=MappingProxyType(
            {
                "OK": SemanticColors("#ffffff", "#007a3d", "#004d27"),
                "NG": SemanticColors("#ffffff", "#c00020", "#7a0014"),
                "NO_PART": SemanticColors("#ffffff", "#a35a00", "#6c3b00"),
                "ERROR": SemanticColors("#ffffff", "#7a0014", "#4a000c"),
                "neutral": SemanticColors("#111827", "#cfd8e3", "#5f6f80"),
            }
        ),
        zone_overlay=ZoneOverlayPalette(
            finalized=(210, 105, 0),
            finalized_fill=(210, 105, 0, 50),
            finalized_label=(80, 40, 0),
            current=(0, 92, 184),
        ),
    ),
    spacing=FACTORY_DARK.spacing,
    typography=FACTORY_DARK.typography,
    dimensions=replace(FACTORY_DARK.dimensions, radius=2, input_radius=2, scrollbar_radius=2),
)

THEME_REGISTRY: Mapping[str, InspectionTheme] = MappingProxyType(
    {
        FACTORY_DARK.name: FACTORY_DARK,
        INDUSTRIAL_LIGHT_DEVCPP.name: INDUSTRIAL_LIGHT_DEVCPP,
        HIGH_CONTRAST_FACTORY_LIGHT.name: HIGH_CONTRAST_FACTORY_LIGHT,
    }
)
DEFAULT_THEME_NAME = INDUSTRIAL_LIGHT_DEVCPP.name
_ACTIVE_THEME = INDUSTRIAL_LIGHT_DEVCPP

THEME_DISPLAY_NAMES: Mapping[str, str] = MappingProxyType(
    {
        FACTORY_DARK.name: "Factory Dark",
        INDUSTRIAL_LIGHT_DEVCPP.name: "Industrial Light",
        HIGH_CONTRAST_FACTORY_LIGHT.name: "High Contrast Light",
    }
)

# Compatibility exports for existing UI modules. Prefer helper functions or a resolved
# InspectionTheme for new code.
APP_BACKGROUND = INDUSTRIAL_LIGHT_DEVCPP.palette.app_background
SHELL_BACKGROUND = INDUSTRIAL_LIGHT_DEVCPP.palette.shell_background
SURFACE = INDUSTRIAL_LIGHT_DEVCPP.palette.surface
SURFACE_ELEVATED = INDUSTRIAL_LIGHT_DEVCPP.palette.surface_elevated
SURFACE_HOVER = INDUSTRIAL_LIGHT_DEVCPP.palette.surface_hover
BORDER = INDUSTRIAL_LIGHT_DEVCPP.palette.border
BORDER_STRONG = INDUSTRIAL_LIGHT_DEVCPP.palette.border_strong
TEXT_PRIMARY = INDUSTRIAL_LIGHT_DEVCPP.palette.text_primary
TEXT_SECONDARY = INDUSTRIAL_LIGHT_DEVCPP.palette.text_secondary
TEXT_MUTED = INDUSTRIAL_LIGHT_DEVCPP.palette.text_muted
ACCENT = INDUSTRIAL_LIGHT_DEVCPP.palette.accent
ACCENT_MUTED = INDUSTRIAL_LIGHT_DEVCPP.palette.accent_muted
SUCCESS = INDUSTRIAL_LIGHT_DEVCPP.palette.success
WARNING = INDUSTRIAL_LIGHT_DEVCPP.palette.warning
DANGER = INDUSTRIAL_LIGHT_DEVCPP.palette.danger
SELECTION = INDUSTRIAL_LIGHT_DEVCPP.palette.selection
PAGE_MARGIN = INDUSTRIAL_LIGHT_DEVCPP.spacing.page_margin_right
SECTION_GAP = INDUSTRIAL_LIGHT_DEVCPP.spacing.section_gap
PANEL_PADDING = INDUSTRIAL_LIGHT_DEVCPP.spacing.panel_padding
CONTROL_GAP = INDUSTRIAL_LIGHT_DEVCPP.spacing.control_gap
RADIUS = INDUSTRIAL_LIGHT_DEVCPP.dimensions.radius


def available_theme_names() -> tuple[str, ...]:
    return tuple(THEME_REGISTRY)


def theme_display_name(theme_name: str) -> str:
    return THEME_DISPLAY_NAMES.get(theme_name, theme_name.replace("_", " ").title())


def resolve_theme(theme_name: str | None = None) -> InspectionTheme:
    name = theme_name or DEFAULT_THEME_NAME
    try:
        return THEME_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(available_theme_names())
        raise ValueError(f"Unknown desktop theme '{name}'. Available themes: {available}") from exc


def active_theme() -> InspectionTheme:
    return _ACTIVE_THEME


def page_margins(theme: InspectionTheme | None = None) -> tuple[int, int, int, int]:
    spacing = (theme or active_theme()).spacing
    return (
        spacing.page_margin_left,
        spacing.page_margin_top,
        spacing.page_margin_right,
        spacing.page_margin_bottom,
    )


def zero_margins() -> tuple[int, int, int, int]:
    return (0, 0, 0, 0)


def group_control_margins(theme: InspectionTheme | None = None) -> tuple[int, int, int, int]:
    spacing = (theme or active_theme()).spacing
    return (
        spacing.group_margin_left,
        spacing.group_margin_top,
        spacing.group_margin_right,
        spacing.group_margin_bottom,
    )


def group_content_margins(theme: InspectionTheme | None = None) -> tuple[int, int, int, int]:
    spacing = (theme or active_theme()).spacing
    return (
        spacing.group_content_margin_left,
        spacing.group_content_margin_top,
        spacing.group_content_margin_right,
        spacing.group_content_margin_bottom,
    )


def theme_spacing(theme: InspectionTheme | None = None) -> ThemeSpacing:
    return (theme or active_theme()).spacing


def theme_dimensions(theme: InspectionTheme | None = None) -> ThemeDimensions:
    return (theme or active_theme()).dimensions


def preview_surface_stylesheet(theme: InspectionTheme | None = None) -> str:
    resolved = theme or active_theme()
    palette = resolved.palette
    return (
        f"background: {palette.canvas_surface}; "
        f"color: {palette.text_secondary}; "
        f"border: 1px solid {palette.border_strong};"
    )


def table_viewport_stylesheet(theme: InspectionTheme | None = None) -> str:
    palette = (theme or active_theme()).palette
    return f"background: {palette.canvas_surface};"


def status_colors(level: str, theme: InspectionTheme | None = None) -> SemanticColors:
    palette = (theme or active_theme()).palette
    return palette.status.get(level, palette.status["info"])


def result_colors(result: str, theme: InspectionTheme | None = None) -> SemanticColors:
    palette = (theme or active_theme()).palette
    return palette.result.get(result, palette.result["neutral"])


def zone_overlay_palette(theme: InspectionTheme | None = None) -> ZoneOverlayPalette:
    return (theme or active_theme()).palette.zone_overlay


def _style_asset_url(filename: str) -> str:
    return (Path(__file__).resolve().parents[2] / "assets" / "ui" / filename).as_posix()


def build_app_stylesheet(theme: InspectionTheme | None = None) -> str:
    resolved = theme or active_theme()
    p = resolved.palette
    s = resolved.spacing
    t = resolved.typography
    d = resolved.dimensions
    info = status_colors("info", resolved)
    success = status_colors("success", resolved)
    warning = status_colors("warning", resolved)
    error = status_colors("error", resolved)
    result_ok = result_colors("OK", resolved)
    result_ng = result_colors("NG", resolved)
    result_no_part = result_colors("NO_PART", resolved)
    result_error = result_colors("ERROR", resolved)
    result_neutral = result_colors("neutral", resolved)
    checkbox_check_icon = _style_asset_url("checkbox_check.svg")
    checkbox_disabled_check_icon = _style_asset_url("checkbox_check_disabled.svg")
    radio_dot_icon = _style_asset_url("radio_dot.svg")
    radio_disabled_dot_icon = _style_asset_url("radio_dot_disabled.svg")
    spinbox_arrow_variant = "on_dark" if resolved.name == "factory_dark" else "on_light"
    spinbox_up_arrow_icon = _style_asset_url(f"spinbox_arrow_up_{spinbox_arrow_variant}.svg")
    spinbox_down_arrow_icon = _style_asset_url(f"spinbox_arrow_down_{spinbox_arrow_variant}.svg")
    spinbox_arrow_disabled_icon = _style_asset_url("spinbox_arrow_disabled.svg")

    return f"""
    QMainWindow {{
        background: {p.app_background};
        color: {p.text_primary};
    }}
    QWidget {{
        background: {p.app_background};
        color: {p.text_primary};
        selection-background-color: {p.selection};
        selection-color: {p.selection_text};
    }}
    QWidget#contentHost,
    QStackedWidget {{
        background: {p.app_background};
    }}
    QScrollArea {{
        background: {p.app_background};
        border: 0;
    }}
    QScrollArea > QWidget > QWidget {{
        background: {p.app_background};
    }}
    QLabel {{
        color: {p.text_primary};
        background: transparent;
    }}
    QLabel#pageHeaderTitle {{
        color: {p.text_primary};
        font-size: {t.page_title_px}px;
        font-weight: 650;
    }}
    QLabel#pageHeaderSubtitle {{
        color: {p.text_secondary};
        font-size: {t.page_subtitle_px}px;
    }}
    QWidget#sectionPanel {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {d.radius}px;
    }}
    QWidget#sectionPanel[styleVariant="classic"] {{
        background: {p.surface_elevated};
        border: 1px solid {p.border_strong};
    }}
    QLabel#sectionPanelTitle {{
        color: {p.text_primary};
        font-size: {t.section_title_px}px;
        font-weight: 700;
    }}
    QWidget#sectionPanel[density="compact"] QLabel#sectionPanelTitle {{
        font-size: {t.section_title_compact_px}px;
    }}
    QLabel#sectionPanelSubtitle {{
        color: {p.text_secondary};
        font-size: {t.section_subtitle_px}px;
    }}
    QWidget#sectionPanel[density="compact"] QLabel#sectionPanelSubtitle {{
        font-size: {t.section_subtitle_compact_px}px;
    }}
    QFrame#statusBanner {{
        border-radius: {d.radius}px;
        font-size: {t.status_px}px;
    }}
    QLabel#statusBannerText,
    QLabel#semanticIcon {{
        background: transparent;
    }}
    QFrame#statusBanner[level="info"] {{
        color: {info.foreground};
        background: {info.background};
        border: 1px solid {info.border};
    }}
    QFrame#statusBanner[level="info"] QLabel {{
        color: {info.foreground};
    }}
    QFrame#statusBanner[level="success"] {{
        color: {success.foreground};
        background: {success.background};
        border: 1px solid {success.border};
    }}
    QFrame#statusBanner[level="success"] QLabel {{
        color: {success.foreground};
    }}
    QFrame#statusBanner[level="warning"] {{
        color: {warning.foreground};
        background: {warning.background};
        border: 1px solid {warning.border};
    }}
    QFrame#statusBanner[level="warning"] QLabel {{
        color: {warning.foreground};
    }}
    QFrame#statusBanner[level="error"] {{
        color: {error.foreground};
        background: {error.background};
        border: 1px solid {error.border};
    }}
    QFrame#statusBanner[level="error"] QLabel {{
        color: {error.foreground};
    }}
    QFrame#statusBanner[state="processing"],
    QFrame#statusBanner[state="live"] {{
        font-weight: 650;
    }}
    QWidget#pathPickerRow {{
        background: transparent;
    }}
    QLabel#pathPickerLabel {{
        color: {p.text_secondary};
    }}
    QWidget#actionButtonRow {{
        background: transparent;
    }}
    QWidget#metricGrid {{
        background: transparent;
    }}
    QLabel#metricLabel {{
        color: {p.text_secondary};
        font-size: {t.metric_label_px}px;
    }}
    QLabel#metricValue {{
        color: {p.text_primary};
        font-size: {t.metric_value_px}px;
        font-weight: 600;
    }}
    QFrame#resultBadge {{
        color: {p.text_on_result};
        border-radius: {d.radius}px;
        font-size: {t.result_badge_px}px;
        font-weight: 750;
    }}
    QFrame#resultBadge QLabel {{
        color: {p.text_on_result};
        font-size: {t.result_badge_px}px;
        font-weight: 750;
    }}
    QFrame#resultBadge[result="OK"] {{
        background: {result_ok.background};
        border: 1px solid {result_ok.border};
    }}
    QFrame#resultBadge[result="NG"] {{
        background: {result_ng.background};
        border: 1px solid {result_ng.border};
    }}
    QFrame#resultBadge[result="NO_PART"] {{
        background: {result_no_part.background};
        border: 1px solid {result_no_part.border};
    }}
    QFrame#resultBadge[result="ERROR"] {{
        background: {result_error.background};
        border: 1px solid {result_error.border};
    }}
    QFrame#resultBadge[result="neutral"] {{
        background: {result_neutral.background};
        border: 1px solid {result_neutral.border};
    }}
    QFrame#resultBadge[result="neutral"] QLabel {{
        color: {p.text_primary};
    }}
    QLabel#resultDetailLabel {{
        color: {p.text_secondary};
        font-size: {t.metric_label_px}px;
        font-weight: 650;
    }}
    QPlainTextEdit#resultDetailText {{
        color: {p.text_primary};
        background: {p.surface_inset};
        border: 1px solid {p.border};
        border-radius: {d.input_radius}px;
        padding: 6px 8px;
    }}
    QWidget#resultImageTabs {{
        background: transparent;
    }}
    QWidget#inspectImageLeftPane,
    QWidget#setupConsole,
    QWidget#setupConsoleColumn,
    QWidget#logsReviewToolbar,
    QWidget#logsRecordsPane,
    QWidget#logsReviewPane,
    QScrollArea#inspectImageLeftPaneScroll,
    QScrollArea#cameraControlPaneScroll,
    QScrollArea#referenceCaptureControlPaneScroll,
    QScrollArea#zoneEditorControlPaneScroll,
    QScrollArea#logsSelectedRecordScroll,
    QWidget#referenceCaptureControlPane,
    QWidget#zoneEditorControlPane,
    QWidget#cameraControlPane,
    QWidget#cameraPreviewContainer,
    QWidget#cameraActionsArea,
    QWidget#operatorReviewArea,
    QWidget#logsReviewActions,
    QWidget#logsSelectedRecordContext,
    QWidget#logsSelectedRecordSummary,
    QWidget#cameraOperatorWorkspace,
    QSplitter#inspectImageWorkspace,
    QSplitter#referenceCaptureWorkspace,
    QSplitter#zoneEditorWorkspace,
    QSplitter#cameraOperatorWorkspace,
    QWidget#selectedRecordReview {{
        background: transparent;
    }}
    QSplitter#logsReviewSplitter,
    QStackedWidget#cameraVisualStack {{
        background: transparent;
    }}
    QScrollArea#selectedRecordDetailsScroll {{
        background: transparent;
        border: 0;
    }}
    QScrollArea#inspectImageLeftPaneScroll > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollArea#cameraControlPaneScroll > QWidget > QWidget,
    QScrollArea#referenceCaptureControlPaneScroll > QWidget > QWidget,
    QScrollArea#zoneEditorControlPaneScroll > QWidget > QWidget,
    QScrollArea#logsSelectedRecordScroll > QWidget > QWidget {{
        background: transparent;
    }}
    QLabel#logsEmptyReview {{
        color: {p.text_secondary};
        background: {p.surface_inset};
        border: 1px solid {p.border};
        border-radius: {d.radius}px;
        padding: 10px 12px;
    }}
    QWidget#logsSummaryGrid,
    QWidget#logsMoreDetails,
    QWidget#resultTechnicalDetails {{
        background: transparent;
    }}
    QWidget#presenceTuningFields {{
        background: {p.surface_inset};
        border: 1px solid {p.border};
        border-radius: {d.input_radius}px;
        padding: 8px;
    }}
    QLabel#logsSourceSummary,
    QLabel#logsRecordIdentity,
    QLabel#logsRecordTime,
    QLabel#logsRecordContext {{
        color: {p.text_secondary};
        background: transparent;
    }}
    QLabel#logsRecordIdentity {{
        color: {p.text_primary};
        font-weight: 650;
    }}
    QPushButton#moreDetailsButton {{
        color: {p.text_secondary};
        background: {p.surface};
        border-color: {p.border};
        text-align: left;
        padding: 6px 10px;
    }}
    QPushButton#moreDetailsButton:checked {{
        color: {p.text_primary};
        background: {p.surface_elevated};
        border-color: {p.border_strong};
    }}
    QGraphicsView#imagePreviewWidget {{
        background: {p.surface_elevated};
        color: {p.text_secondary};
        border: 1px solid {p.border_strong};
    }}
    QTableWidget#recordsTable {{
        background: {p.surface_elevated};
        alternate-background-color: {p.table_alternate};
        color: {p.table_text};
        border: 1px solid {p.border_strong};
        selection-background-color: {p.selection};
        selection-color: {p.selection_text};
    }}
    QTableWidget#recordsTable::item {{
        padding: 4px 8px;
        border: 0;
    }}
    QTableWidget#recordsTable::item:selected {{
        background: {p.selection};
        color: {p.selection_text};
    }}
    QHeaderView::section {{
        background: {p.table_header};
        color: {p.text_primary};
        font-weight: 600;
        padding: 6px 8px;
        border: 0;
        border-right: 1px solid {p.border_strong};
        border-bottom: 1px solid {p.border_strong};
    }}
    QTableCornerButton::section {{
        background: {p.table_header};
        border: 0;
        border-right: 1px solid {p.border_strong};
        border-bottom: 1px solid {p.border_strong};
    }}
    QSplitter::handle {{
        background: {p.border};
    }}
    QFrame#topNavShell {{
        background: {p.shell_background};
        border-bottom: 1px solid {p.border};
    }}
    QFrame#topNavShell[compact="true"] {{
        border-bottom: 1px solid {p.border_strong};
    }}
    QWidget#topPageContextArea,
    QWidget#topBrandArea,
    QWidget#topNavCluster,
    QWidget#topNavGroup,
    QWidget#topThemeArea {{
        background: transparent;
    }}
    QLabel#topPageContextTitle {{
        color: {p.text_primary};
        font-size: {t.section_title_px}px;
        font-weight: 700;
    }}
    QLabel#topPageContextSubtitle {{
        color: {p.text_secondary};
        font-size: {t.shell_subtitle_px}px;
    }}
    QLabel#topNavTitle {{
        color: {p.text_primary};
        font-size: {t.section_title_px}px;
        font-weight: 700;
    }}
    QLabel#topNavSubtitle {{
        color: {p.text_muted};
        font-size: {t.shell_subtitle_px}px;
    }}
    QWidget#topNavGroup {{
        border: 0;
        padding: 0;
    }}
    QLabel#navGroupLabel {{
        color: {p.text_muted};
        font-size: {t.nav_group_px}px;
        font-weight: 700;
        letter-spacing: 0;
        padding: 0 4px 0 4px;
    }}
    QFrame#topNavSeparator {{
        color: {p.border};
        background: {p.border};
        max-width: 1px;
        margin: 5px 6px;
    }}
    QPushButton#navButton {{
        min-height: {d.nav_button_min_height}px;
        padding: 4px 9px;
        border: 1px solid transparent;
        border-radius: {d.radius}px;
        color: {p.text_secondary};
        background: transparent;
    }}
    QFrame#topNavShell[compact="true"] QPushButton#navButton {{
        padding: 4px 7px;
    }}
    QPushButton#navButton:hover {{
        color: {p.text_primary};
        background: {p.surface};
        border-color: {p.accent};
    }}
    QPushButton#navButton:focus {{
        color: {p.text_primary};
        background: {p.surface};
        border-color: {p.accent};
    }}
    QPushButton#navButton:checked {{
        color: {p.text_primary};
        background: {p.accent_muted};
        border-color: {p.accent};
        font-weight: 650;
    }}
    QPushButton#fitImageButton {{
        min-height: 0;
        padding: 7px 12px;
        margin: 0 0 0 6px;
        border: 1px solid {p.border};
        border-bottom-color: {p.border_strong};
        border-radius: 0;
        color: {p.text_secondary};
        background: {p.surface};
    }}
    QPushButton#fitImageButton:hover {{
        color: {p.text_primary};
        background: {p.accent_muted};
        border-color: {p.accent};
    }}
    QPushButton#fitImageButton:focus {{
        color: {p.text_primary};
        background: {p.surface_elevated};
        border-color: {p.accent};
    }}
    QTabWidget#resultImageTabWidget QTabBar::tab {{
        min-height: 0;
    }}
    QGroupBox {{
        color: {p.text_primary};
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {d.radius}px;
        margin-top: 12px;
        padding: 12px;
        font-weight: 650;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: {p.text_primary};
        background: {p.app_background};
    }}
    QGroupBox#logsSourceControls,
    QGroupBox#logsFilterControls {{
        padding: 8px;
        margin-top: 10px;
    }}
    QGroupBox#logsSourceControls::title,
    QGroupBox#logsFilterControls::title {{
        left: 8px;
    }}
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPlainTextEdit {{
        color: {p.text_primary};
        background: {p.surface_elevated};
        border: 1px solid {p.border_strong};
        border-radius: {d.input_radius}px;
        padding: 6px 8px;
        selection-background-color: {p.selection};
    }}
    QSpinBox,
    QDoubleSpinBox {{
        padding: 6px 30px 6px 8px;
    }}
    QSpinBox::up-button,
    QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 24px;
        border-left: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        border-top-right-radius: {d.input_radius}px;
        background: {p.surface_elevated};
        margin: 1px 1px 0 0;
    }}
    QSpinBox::down-button,
    QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 24px;
        border-left: 1px solid {p.border};
        border-top: 1px solid {p.border};
        border-bottom-right-radius: {d.input_radius}px;
        background: {p.surface_elevated};
        margin: 0 1px 1px 0;
    }}
    QSpinBox::up-button:hover,
    QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover,
    QDoubleSpinBox::down-button:hover {{
        background: {p.accent_muted};
        border-left-color: {p.accent};
    }}
    QSpinBox::up-button:disabled,
    QDoubleSpinBox::up-button:disabled,
    QSpinBox::down-button:disabled,
    QDoubleSpinBox::down-button:disabled {{
        background: {p.surface_disabled};
        border-color: {p.border};
    }}
    QSpinBox::up-arrow,
    QDoubleSpinBox::up-arrow {{
        width: 9px;
        height: 9px;
        image: url("{spinbox_up_arrow_icon}");
    }}
    QSpinBox::down-arrow,
    QDoubleSpinBox::down-arrow {{
        width: 9px;
        height: 9px;
        image: url("{spinbox_down_arrow_icon}");
    }}
    QSpinBox::up-arrow:disabled,
    QDoubleSpinBox::up-arrow:disabled,
    QSpinBox::down-arrow:disabled,
    QDoubleSpinBox::down-arrow:disabled {{
        image: url("{spinbox_arrow_disabled_icon}");
    }}
    QLineEdit:read-only {{
        color: {p.text_secondary};
        background: {p.surface_inset};
    }}
    QLineEdit:focus,
    QSpinBox:focus,
    QDoubleSpinBox:focus,
    QComboBox:focus,
    QPlainTextEdit:focus {{
        border-color: {p.accent};
        background: {p.surface_elevated};
    }}
    QLineEdit:hover,
    QSpinBox:hover,
    QDoubleSpinBox:hover,
    QComboBox:hover,
    QPlainTextEdit:hover {{
        border-color: {p.accent};
        background: {p.surface_hover};
    }}
    QLineEdit:disabled,
    QSpinBox:disabled,
    QDoubleSpinBox:disabled,
    QComboBox:disabled,
    QPlainTextEdit:disabled {{
        color: {p.text_muted};
        background: {p.surface_disabled};
        border-color: {p.border};
    }}
    QComboBox::drop-down {{
        border: 0;
        width: 24px;
    }}
    QComboBox#themeSelector {{
        padding-left: 6px;
        padding-right: 4px;
    }}
    QCheckBox,
    QRadioButton {{
        color: {p.text_primary};
        spacing: {s.control_gap}px;
        background: transparent;
    }}
    QCheckBox:disabled,
    QRadioButton:disabled {{
        color: {p.text_muted};
    }}
    QCheckBox::indicator,
    QRadioButton::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid {p.border_strong};
        background: {p.surface_elevated};
    }}
    QCheckBox::indicator {{
        border-radius: 3px;
    }}
    QRadioButton::indicator {{
        border-radius: 7px;
    }}
    QCheckBox::indicator:hover,
    QRadioButton::indicator:hover {{
        border-color: {p.accent};
        background: {p.surface_hover};
    }}
    QCheckBox:focus::indicator,
    QRadioButton:focus::indicator {{
        border-color: {p.accent};
        background: {p.surface_hover};
    }}
    QCheckBox::indicator:checked,
    QCheckBox::indicator:checked:hover {{
        border-color: {p.accent_button_hover_border};
        background: {p.accent_button};
        image: url("{checkbox_check_icon}");
    }}
    QRadioButton::indicator:checked {{
        border-color: {p.accent};
        background: {p.accent_button};
        image: url("{radio_dot_icon}");
    }}
    QRadioButton::indicator:checked:hover {{
        border-color: {p.accent_button_hover_border};
        background: {p.accent_button_hover};
        image: url("{radio_dot_icon}");
    }}
    QCheckBox::indicator:disabled,
    QRadioButton::indicator:disabled {{
        border-color: {p.border};
        background: {p.surface_disabled};
    }}
    QCheckBox::indicator:checked:disabled {{
        border-color: {p.border_strong};
        background: {p.surface_disabled};
        image: url("{checkbox_disabled_check_icon}");
    }}
    QRadioButton::indicator:checked:disabled {{
        border-color: {p.border_strong};
        background: {p.surface_disabled};
        image: url("{radio_disabled_dot_icon}");
    }}
    QPushButton {{
        color: {p.text_primary};
        background: {p.surface_elevated};
        border: 1px solid {p.border_strong};
        border-radius: {d.radius}px;
        padding: 7px 12px;
        min-height: {d.button_min_height}px;
    }}
    QPushButton:hover {{
        background: {p.accent_muted};
        border-color: {p.accent};
    }}
    QPushButton:focus {{
        background: {p.accent_muted};
        border-color: {p.accent};
    }}
    QPushButton:pressed {{
        background: {p.accent_muted};
    }}
    QPushButton:disabled {{
        color: {p.text_muted};
        background: {p.surface_disabled};
        border-color: {p.border};
    }}
    QPushButton:disabled:hover {{
        color: {p.text_muted};
        background: {p.surface_disabled};
        border-color: {p.border};
    }}
    QPushButton[buttonRole="primary"] {{
        color: {p.text_on_accent};
        background: {p.accent_button};
        border-color: {p.accent};
        font-weight: 650;
    }}
    QPushButton[buttonRole="primary"]:hover {{
        background: {p.accent_button_hover};
        border-color: {p.accent_button_hover_border};
    }}
    QPushButton[buttonRole="primary"]:focus {{
        background: {p.accent_button_hover};
        border-color: {p.accent_button_hover_border};
    }}
    QPushButton[buttonRole="secondary"] {{
        color: {p.text_primary};
        background: {p.surface_elevated};
        border-color: {p.border_strong};
    }}
    QPushButton[buttonRole="secondary"]:hover {{
        color: {p.text_primary};
        background: {p.accent_muted};
        border-color: {p.accent};
    }}
    QPushButton[buttonRole="secondary"]:focus {{
        color: {p.text_primary};
        background: {p.accent_muted};
        border-color: {p.accent};
    }}
    QPushButton[buttonRole="danger"] {{
        color: {p.danger_button_foreground};
        background: {p.danger_button_background};
        border-color: {p.danger};
    }}
    QPushButton[buttonRole="danger"]:hover,
    QPushButton[buttonRole="danger"]:focus {{
        color: {p.danger_button_foreground};
        background: {p.danger};
        border-color: {p.danger};
    }}
    QPushButton[buttonRole="primary"]:disabled,
    QPushButton[buttonRole="secondary"]:disabled,
    QPushButton[buttonRole="danger"]:disabled,
    QPushButton[buttonRole="primary"]:disabled:hover,
    QPushButton[buttonRole="secondary"]:disabled:hover,
    QPushButton[buttonRole="danger"]:disabled:hover {{
        color: {p.text_muted};
        background: {p.surface_disabled};
        border-color: {p.border};
    }}
    QTabWidget::pane {{
        border: 1px solid {p.border_strong};
        background: {p.surface};
        top: -1px;
    }}
    QTabBar::tab {{
        color: {p.text_secondary};
        background: {p.surface};
        border: 1px solid {p.border};
        border-bottom: 0;
        padding: 8px 12px;
        margin-right: 2px;
    }}
    QTabBar::tab:hover {{
        color: {p.text_primary};
        background: {p.surface_hover};
        border-color: {p.accent};
    }}
    QTabBar::tab:selected {{
        color: {p.text_primary};
        background: {p.surface_elevated};
        border-color: {p.accent};
    }}
    QScrollBar:vertical {{
        background: {p.app_background};
        width: {d.scrollbar_size}px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        border-radius: {d.scrollbar_radius}px;
        min-height: 28px;
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {p.app_background};
        height: {d.scrollbar_size}px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.border_strong};
        border-radius: {d.scrollbar_radius}px;
        min-width: 28px;
    }}
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    """


def apply_app_theme(app: QApplication, theme_name: str | None = None) -> InspectionTheme:
    global _ACTIVE_THEME
    theme = resolve_theme(theme_name)
    _ACTIVE_THEME = theme
    app.setStyleSheet(build_app_stylesheet(theme))
    return theme
