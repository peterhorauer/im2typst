"""Synthetic Typst-math generator: AST nodes + a depth-bounded random sampler."""

from .grammar import Grammar
from . import nodes

__all__ = ["Grammar", "nodes"]
