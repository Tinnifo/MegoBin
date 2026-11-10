"""Contracts: every stage says what it NEEDS and what it PROVIDES.

A *port* is a named piece of data (``features``, ``z``, ``bins``, ``graph``).
Its *content* is a set of labels (``mu``, ``var``, ``kmer``, ``depth``,
``views=2``, ``assembly_graph`` ...). A stage declares ``needs`` and
``provides`` as ``{port: {labels}}``.

* :func:`resolve` builds the stage order backwards from the requested target
  port (Snakemake-style). Stages nobody needs are not run.
* :func:`check` walks the resolved order and compares labels, including the
  inner contracts of stages that contain a training loop. Nothing has been
  computed when it runs.

A failure names the component, the port and the missing labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

Labels = frozenset
Needs = Mapping[str, "frozenset[str]"]


def labels(*names: str) -> frozenset[str]:
    """Shorthand for a label set: ``labels("mu", "var")``."""
    return frozenset(names)


def as_labels(items: Iterable[str] | None) -> frozenset[str]:
    return frozenset(items or ())


@dataclass(frozen=True)
class Problem:
    """One contract violation.

    ``missing`` empty means the whole port is unavailable; otherwise the port
    exists but lacks these labels.
    """

    component: str
    port: str
    missing: tuple[str, ...] = ()
    offered: tuple[str, ...] = ()
    provider: str = ""
    detail: str = ""

    def __str__(self) -> str:
        if self.detail and not self.missing and not self.offered:
            return f"{self.component}: {self.detail}"
        if not self.provider and not self.offered:
            where = "no stage provides it and the input data does not offer it"
            want = f" with labels {list(self.missing)}" if self.missing else ""
            return f"{self.component}: needs port '{self.port}'{want}, but {where}"
        by = f" by {self.provider}" if self.provider else ""
        return (
            f"{self.component}: needs {self.port}.{list(self.missing)}; "
            f"offered {self.port}.{list(self.offered)}{by}"
        )


class ContractError(ValueError):
    """Raised when one or more contracts fail. ``problems`` lists them all."""

    def __init__(self, problems: list[Problem] | list[str]):
        self.problems: list[Problem] = [p for p in problems if isinstance(p, Problem)]
        self.messages: list[str] = [str(p) for p in problems]
        super().__init__("; ".join(self.messages))


class Stage:
    """One node of the method DAG.

    Subclasses set ``needs`` / ``provides`` (``{port: labels}``) in ``__init__``
    or in :meth:`bind`, which the pipeline calls once with the manifest of
    available input ports, before resolution. ``check_inner`` lets a stage that
    contains an inner loop (training) check its own sub-components.
    """

    name: str = ""

    def __init__(
        self,
        needs: Mapping[str, Iterable[str]] | None = None,
        provides: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.needs: dict[str, frozenset[str]] = {
            k: frozenset(v) for k, v in (needs or {}).items()
        }
        self.provides: dict[str, frozenset[str]] = {
            k: frozenset(v) for k, v in (provides or {}).items()
        }

    def bind(self, manifest: Any) -> None:
        """Adjust needs/provides to the available input ports (default: no-op)."""

    def check_inner(self) -> list[Problem]:
        return []

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        raise NotImplementedError


def resolve(
    stages: Mapping[str, Stage],
    target: str,
    available: Mapping[str, Iterable[str]],
) -> list[str]:
    """Backward resolution from ``target``. Returns the stage run order.

    Raises :class:`ContractError` if a needed port has no provider (and is not
    offered by the input data), if two stages provide the same port, or on a
    cycle.
    """
    providers: dict[str, list[str]] = {}
    for name, st in stages.items():
        for port in st.provides:
            providers.setdefault(port, []).append(name)

    order: list[str] = []
    visiting: set[str] = set()
    problems: list[Problem] = []

    def need(port: str, asked_by: str, wanted: Iterable[str] = ()) -> None:
        if port in available:
            return
        cands = providers.get(port, [])
        if not cands:
            problems.append(Problem(asked_by, port, tuple(sorted(wanted))))
            return
        if len(cands) > 1:
            problems.append(
                Problem(
                    asked_by,
                    port,
                    detail=f"port '{port}' is provided by several stages {cands}; "
                    "rename one port so the DAG is unambiguous",
                )
            )
            return
        visit(cands[0])

    def visit(name: str) -> None:
        if name in order:
            return
        if name in visiting:
            problems.append(Problem(name, "", detail=f"cycle through stage '{name}'"))
            return
        visiting.add(name)
        for port, wanted in stages[name].needs.items():
            need(port, name, wanted)
        visiting.discard(name)
        order.append(name)

    need(target, "the pipeline target")
    if problems:
        raise ContractError(problems)
    return order


def check(
    stages: Mapping[str, Stage],
    order: list[str],
    available: Mapping[str, Iterable[str]],
) -> list[Problem]:
    """Pipeline-start check: walk ``order`` and compare labels.

    Returns a list of problems (empty = OK).
    """
    offered: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in available.items()}
    source: dict[str, str] = {k: "the input data" for k in available}
    problems: list[Problem] = []
    for name in order:
        st = stages[name]
        problems += st.check_inner()
        for port, wanted in st.needs.items():
            missing = frozenset(wanted) - offered.get(port, frozenset())
            if missing:
                problems.append(
                    Problem(
                        name,
                        port,
                        tuple(sorted(missing)),
                        tuple(sorted(offered.get(port, ()))),
                        source.get(port, ""),
                    )
                )
        for port, provided in st.provides.items():
            offered[port] = frozenset(provided)
            source[port] = name
    return problems


def bind_kwargs(
    bindings: Mapping[str, str], values: Mapping[str, Any], who: str
) -> dict[str, Any]:
    """Map constructor kwargs to named runtime values.

    ``bindings`` is ``{kwarg: source_name}``, declared on the component class.
    This replaces introspecting ``__init__`` signatures: the component states
    which values it takes, and a missing source is a loud error.
    """
    out: dict[str, Any] = {}
    for kwarg, src in bindings.items():
        if src not in values:
            raise ContractError(
                [Problem(who, src, detail=f"kwarg '{kwarg}' is bound to '{src}', which is not available")]
            )
        out[kwarg] = values[src]
    return out
