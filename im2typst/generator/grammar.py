"""Depth-bounded random sampler that assembles ``nodes`` into Typst math trees.

The grammar is recursive: ``_expr(depth)`` picks a production (binop, fraction,
power, function, big operator, …) and recurses with ``depth - 1`` until it
bottoms out in an :class:`~im2typst.generator.nodes.Atom`. Because every node
emits guaranteed-valid Typst, every sampled tree is valid by construction — the
whole point of generating our own data.

Usage::

    g = Grammar(max_depth=4, seed=0)
    node = g.sample()
    print(node.typst())
"""

from __future__ import annotations

import random

from . import nodes

# --- Terminal vocabularies ---------------------------------------------------

VARIABLES = list("abcdefghkmnpqrstuvwxyz")
DIGITS = [str(d) for d in range(1, 10)]            # single non-zero digits
GREEK = [
    "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
    "mu", "nu", "pi", "rho", "sigma", "tau", "phi", "psi", "omega",
]
CONSTANTS = ["pi", "e", "infinity"]

FUNCS = ["sin", "cos", "tan", "tanh", "log", "ln", "exp", "arcsin", "arccos"]

# Infix operators; "" means implicit multiplication by adjacency (``a b``).
BINOPS = ["+", "-", "", "", "dot", "times", "plus.minus"]
RELOPS = ["=", "<=", ">=", "!=", "<", ">"]

INDEX_VARS = ["i", "j", "k", "n"]
BOUND_UPPER = ["n", "m", "N", "infinity"]


class Grammar:
    """Random sampler for Typst math expression trees."""

    def __init__(self, max_depth: int = 4, seed: int | None = None) -> None:
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    # -- public API -----------------------------------------------------------

    def sample(self) -> nodes.Node:
        """Sample one expression tree.

        With some probability the whole thing is framed as a relation
        (``lhs <rel> rhs``), which looks more like "real" formulas.
        """
        expr = self._expr(self.max_depth)
        if self.rng.random() < 0.35:
            rel = self.rng.choice(RELOPS)
            lhs = self._simple_lhs()
            return nodes.BinOp(rel, lhs, expr)
        return expr

    def sample_typst(self) -> str:
        """Convenience: sample a tree and return its Typst string."""
        return self.sample().typst()

    # -- production rules -----------------------------------------------------

    def _expr(self, depth: int) -> nodes.Node:
        if depth <= 1:
            return self._atom()
        production = self._weighted([
            (self._binop, 5),
            (self._frac, 3),
            (self._power, 3),
            (self._subscript, 2),
            (self._sqrt, 2),
            (self._root, 1),
            (self._func, 3),
            (self._bigop, 2),
            (self._atom, 3),
        ])
        return production(depth)

    def _binop(self, depth: int) -> nodes.Node:
        return nodes.BinOp(
            self.rng.choice(BINOPS),
            self._expr(depth - 1),
            self._expr(depth - 1),
        )

    def _frac(self, depth: int) -> nodes.Node:
        return nodes.Frac(self._expr(depth - 1), self._expr(depth - 1))

    def _power(self, depth: int) -> nodes.Node:
        return nodes.Power(self._expr(depth - 1), self._script(depth))

    def _subscript(self, depth: int) -> nodes.Node:
        return nodes.Subscript(self._expr(depth - 1), self._script(depth))

    def _sqrt(self, depth: int) -> nodes.Node:
        return nodes.Sqrt(self._expr(depth - 1))

    def _root(self, depth: int) -> nodes.Node:
        return nodes.Root(nodes.Atom(self.rng.choice(DIGITS)), self._expr(depth - 1))

    def _func(self, depth: int) -> nodes.Node:
        return nodes.Func(self.rng.choice(FUNCS), self._expr(depth - 1))

    def _bigop(self, depth: int) -> nodes.Node:
        op = self.rng.choice(["sum", "product", "integral"])
        if op == "integral":
            var = self.rng.choice(VARIABLES)
            lower = nodes.Atom(self.rng.choice(VARIABLES + DIGITS))
            upper = nodes.Atom(self.rng.choice(VARIABLES + DIGITS))
            return nodes.BigOp(
                "integral", lower, upper, self._expr(depth - 1),
                differential=f"dif {var}",
            )
        idx = self.rng.choice(INDEX_VARS)
        lower = nodes.BinOp("=", nodes.Atom(idx), nodes.Atom(self.rng.choice(DIGITS)))
        upper = nodes.Atom(self.rng.choice(BOUND_UPPER))
        return nodes.BigOp(op, lower, upper, self._expr(depth - 1))

    # -- terminals & helpers --------------------------------------------------

    def _atom(self, depth: int | None = None) -> nodes.Node:
        r = self.rng.random()
        if r < 0.55:
            return nodes.Atom(self.rng.choice(VARIABLES))
        if r < 0.80:
            return nodes.Atom(self.rng.choice(DIGITS))
        if r < 0.95:
            return nodes.Atom(self.rng.choice(GREEK))
        return nodes.Atom(self.rng.choice(CONSTANTS))

    def _script(self, depth: int) -> nodes.Node:
        """Exponent / subscript: usually a single atom, occasionally an expr."""
        if self.rng.random() < 0.75:
            return self._atom()
        return self._expr(depth - 1)

    def _simple_lhs(self) -> nodes.Node:
        """A short left-hand side for a relation: a var, maybe subscripted."""
        base = nodes.Atom(self.rng.choice(VARIABLES))
        if self.rng.random() < 0.3:
            return nodes.Subscript(base, nodes.Atom(self.rng.choice(INDEX_VARS)))
        return base

    def _weighted(self, choices):
        fns, weights = zip(*choices)
        return self.rng.choices(fns, weights=weights, k=1)[0]
