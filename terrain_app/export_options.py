"""Which pipeline artifacts to build (preview, print, AMS)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExportOptions:
    preview_glb: bool = False
    preview_obj: bool = False
    quad_mask: bool = False
    print_stl: bool = True
    print_3mf: bool = False
    print_textured_glb: bool = False
    print_ams: bool = True
    print_ams_glb: bool = False
    print_pieces: bool = False

    def needs_print_solid(self) -> bool:
        return (
            self.print_stl
            or self.print_3mf
            or self.print_textured_glb
            or self.print_ams
            or self.print_ams_glb
            or self.print_pieces
        )

    def needs_ams(self) -> bool:
        return self.print_ams or self.print_ams_glb

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def _form_bool(form: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = form.get(key)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("", "0", "false", "off", "no"):
        return False
    if s in ("1", "true", "on", "yes"):
        return True
    return default


def parse_export_options(form: Mapping[str, Any] | None) -> ExportOptions:
    """Parse multipart form export checkboxes; STL + AMS on by default."""
    if form is None:
        return ExportOptions()
    return ExportOptions(
        preview_glb=_form_bool(form, "export_preview_glb", False),
        preview_obj=_form_bool(form, "export_preview_obj", False),
        quad_mask=_form_bool(form, "export_quad_mask", False),
        print_stl=_form_bool(form, "export_print_stl", True),
        print_3mf=_form_bool(form, "export_print_3mf", False),
        print_textured_glb=_form_bool(form, "export_print_textured_glb", False),
        print_ams=_form_bool(form, "export_print_ams", True),
        print_ams_glb=_form_bool(form, "export_print_ams_glb", False),
        print_pieces=_form_bool(form, "export_print_pieces", False),
    )


AMS_QUALITY_CHOICES = ("high", "medium", "low")


def normalize_ams_quality(value: str | None) -> str:
    s = (value or "medium").lower().strip()
    if s not in AMS_QUALITY_CHOICES:
        return "medium"
    return s
