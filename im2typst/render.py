"""Wrap a Typst math string in the page template and render it to PNG.

This is the bridge from the synthetic generator to actual image files: it takes
a bare math fragment (e.g. ``x = (-b plus.minus sqrt(b^2 - 4 a c)) / (2 a)``),
wraps it in the validated page template from CLAUDE.md, and shells out to the
Typst CLI to produce a tightly-cropped PNG.

Augmentation hooks (fonts, sizes, background tint, noise, rotation) are exposed
as parameters so the dataset script can diversify renders for robustness to real
screenshots — see the ``RenderOptions`` knobs.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Typst binary; override via the ``TYPST_BIN`` env var if not on PATH.
TYPST_BIN = os.environ.get("TYPST_BIN", "typst")

# The page template proven to work in gen_formula_im/formula.typ. ``width/height
# = auto`` makes Typst crop the page tightly to the formula.
_TEMPLATE = (
    "#set page(width: auto, height: auto, margin: {margin}pt, fill: {fill})\n"
    "#set text(size: {size}pt{font})\n"
    "$ {body} $\n"
)


@dataclass
class RenderOptions:
    """Per-render knobs. Defaults reproduce the plain proof-of-concept render."""

    ppi: int = 200
    margin: int = 4
    text_size: int = 11
    font: str | None = None          # e.g. "New Computer Modern Math"
    fill: str = "white"              # page background, any Typst color expr

    def font_clause(self) -> str:
        return f', font: "{self.font}"' if self.font else ""


class TypstRenderError(RuntimeError):
    """Raised when the Typst CLI fails to compile a fragment."""


def wrap(typst_math: str, opts: RenderOptions | None = None) -> str:
    """Return the full Typst document source for a bare math fragment."""
    opts = opts or RenderOptions()
    return _TEMPLATE.format(
        margin=opts.margin,
        fill=opts.fill,
        size=opts.text_size,
        font=opts.font_clause(),
        body=typst_math,
    )


def render_to_png(
    typst_math: str,
    out_path: str | Path,
    opts: RenderOptions | None = None,
    typst_bin: str = TYPST_BIN,
) -> Path:
    """Render a bare math fragment to ``out_path`` as PNG.

    Raises :class:`TypstRenderError` (with the compiler's stderr) on failure, so
    a malformed fragment is loud rather than silently dropped.
    """
    opts = opts or RenderOptions()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = wrap(typst_math, opts)
    with tempfile.NamedTemporaryFile("w", suffix=".typ", delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        proc = subprocess.run(
            [typst_bin, "compile", "--format", "png", "--ppi", str(opts.ppi),
             tmp, str(out_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise TypstRenderError(
                f"typst failed for {typst_math!r}:\n{proc.stderr.strip()}"
            )
    finally:
        os.unlink(tmp)
    return out_path
