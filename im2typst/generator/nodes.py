"""AST node classes for synthetic Typst math expressions.

Each node implements ``.typst()``, returning a guaranteed-valid Typst math
fragment. The composition rules (where parentheses are required to stay
unambiguous) live *here*, so any tree the grammar assembles renders correctly.
See CLAUDE.md → "Typst math syntax notes".

Design note: a fragment is ``atomic`` when it reads as a single unit and can sit
directly as a power/subscript base without surrounding parentheses (e.g. ``x``,
``sqrt(2)``, ``sin(x)``). Non-atomic fragments (sums, fractions, bare binops)
get wrapped by ``_as_base`` when used as a base.
"""

from __future__ import annotations

from dataclasses import dataclass


class Node:
    """Base class. Subclasses return a Typst math fragment from ``.typst()``."""

    #: Reads as a single unit → usable as a power/subscript base unparenthesized.
    atomic: bool = False

    def typst(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _as_base(self) -> str:
        """Render for use as the base of a power/subscript or an operator body."""
        s = self.typst()
        return s if self.atomic else f"({s})"


@dataclass
class Atom(Node):
    """A single leaf token: variable, digit, Greek letter, or constant."""

    value: str
    atomic = True

    def typst(self) -> str:
        return self.value


@dataclass
class Group(Node):
    """An explicitly parenthesized sub-expression: ``(expr)``."""

    child: Node
    atomic = True

    def typst(self) -> str:
        return f"({self.child.typst()})"


@dataclass
class BinOp(Node):
    """Two operands joined by an infix operator.

    ``op`` is the literal Typst infix token: ``"+"``, ``"-"``, ``"="``,
    ``"dot"``, ``"times"``, ``"plus.minus"``, ``"<="`` … or the empty string
    for implicit multiplication by adjacency (``a b`` → "ab").
    """

    op: str
    left: Node
    right: Node

    def typst(self) -> str:
        sep = " " if self.op == "" else f" {self.op} "
        return f"{self.left.typst()}{sep}{self.right.typst()}"


@dataclass
class Frac(Node):
    """A fraction: ``(num) / (den)`` — operands always parenthesized."""

    num: Node
    den: Node

    def typst(self) -> str:
        return f"({self.num.typst()}) / ({self.den.typst()})"


@dataclass
class Power(Node):
    """A power: ``base^(exp)`` — exponent always wrapped, base wrapped if needed."""

    base: Node
    exp: Node

    def typst(self) -> str:
        return f"{self.base._as_base()}^({self.exp.typst()})"


@dataclass
class Subscript(Node):
    """A subscript: ``base_(sub)`` — subscript always wrapped, base if needed."""

    base: Node
    sub: Node

    def typst(self) -> str:
        return f"{self.base._as_base()}_({self.sub.typst()})"


@dataclass
class Sqrt(Node):
    """A square root: ``sqrt(x)``."""

    radicand: Node
    atomic = True

    def typst(self) -> str:
        return f"sqrt({self.radicand.typst()})"


@dataclass
class Root(Node):
    """An n-th root: ``root(n, x)``."""

    degree: Node
    radicand: Node
    atomic = True

    def typst(self) -> str:
        return f"root({self.degree.typst()}, {self.radicand.typst()})"


@dataclass
class Func(Node):
    """A named function applied to one argument: ``sin(x)``, ``log(y)``.

    Function names render upright automatically in Typst math.
    """

    name: str
    arg: Node
    atomic = True

    def typst(self) -> str:
        return f"{self.name}({self.arg.typst()})"


@dataclass
class BigOp(Node):
    """A large operator with optional bounds and an optional differential.

    Examples::

        sum_(i = 1)^(n) (a_i)
        product_(k = 1)^(m) (x)
        integral_(a)^(b) (f) dif x
    """

    op: str                     # "sum", "product", "integral"
    lower: Node | None
    upper: Node | None
    body: Node
    differential: str | None = None   # e.g. "dif x" for integrals

    def typst(self) -> str:
        s = self.op
        if self.lower is not None:
            s += f"_({self.lower.typst()})"
        if self.upper is not None:
            s += f"^({self.upper.typst()})"
        s += f" {self.body._as_base()}"
        if self.differential:
            s += f" {self.differential}"
        return s
