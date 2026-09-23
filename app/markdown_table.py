"""Pure Markdown-table primitives shared by source and presentation paths.

These helpers only preserve table structure. They do not decide which table
is authoritative, infer units, or rewrite source cells; those policies stay
with their respective narrative layers.
"""
from __future__ import annotations

import re


def table_cells(line: str) -> list[str]:
    """Split one Markdown table row without reformatting source values."""

    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_rule(cells: list[str]) -> bool:
    """Whether cells are a Markdown alignment/rule row."""

    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


__all__ = ["is_table_rule", "table_cells"]
