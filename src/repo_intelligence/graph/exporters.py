"""M5 diagram exporters: DiagramProjection → Mermaid / Graphviz DOT.

Both exporters serialize the validated DiagramProjection only. They never
perform architecture inference. Output is byte-for-byte deterministic for the
same projection.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from .models import DiagramProjection


def _escape_mermaid_label(label: str) -> str:
    return str(label).replace('"', "'")


def _escape_dot_label(label: str) -> str:
    return str(label).replace("\\", "\\\\").replace('"', '\\"')


def render_mermaid(projection: DiagramProjection) -> str:
    """Serialize a DiagramProjection as deterministic Mermaid flowchart source."""
    lines: List[str] = ["flowchart TD"]
    for node in projection.nodes:
        lines.append(
            '    {}["{}"]'.format(
                node["diagram_id"], _escape_mermaid_label(node["label"])
            )
        )
    for rel in projection.relationships:
        lines.append(
            "    {} -->|{}| {}".format(
                rel["source_diagram_id"],
                _escape_mermaid_label(rel["label"]),
                rel["target_diagram_id"],
            )
        )
    return "\n".join(lines) + "\n"


def render_graphviz(projection: DiagramProjection) -> str:
    """Serialize a DiagramProjection as deterministic Graphviz DOT source."""
    lines: List[str] = ["digraph architecture {"]
    for node in projection.nodes:
        lines.append(
            '  {} [label="{}"];'.format(
                node["diagram_id"], _escape_dot_label(node["label"])
            )
        )
    for rel in projection.relationships:
        lines.append(
            '  {} -> {} [label="{}"];'.format(
                rel["source_diagram_id"],
                rel["target_diagram_id"],
                _escape_dot_label(rel["label"]),
            )
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_mermaid(
    projection: DiagramProjection, output_path: Optional[Union[Path, str]] = None
) -> str:
    text = render_mermaid(projection)
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    return text


def export_graphviz(
    projection: DiagramProjection, output_path: Optional[Union[Path, str]] = None
) -> str:
    text = render_graphviz(projection)
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    return text


def graphviz_available() -> bool:
    return shutil.which("dot") is not None


def render_svg(dot_path: Union[Path, str], svg_path: Union[Path, str]) -> Optional[Path]:
    """Render DOT → SVG when Graphviz is installed. Returns None otherwise.

    SVG is a presentation artifact only — never a source of truth.
    """
    if not graphviz_available():
        return None
    dot = Path(dot_path)
    svg = Path(svg_path)
    if not dot.is_file():
        return None
    svg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dot", "-Tsvg", str(dot), "-o", str(svg)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return svg


__all__ = [
    "export_graphviz",
    "export_mermaid",
    "graphviz_available",
    "render_graphviz",
    "render_mermaid",
    "render_svg",
]
