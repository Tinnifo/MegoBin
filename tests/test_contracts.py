"""resolve() / check() on toy stages: backward resolution and label checks."""

import pytest

from megobin.contracts import ContractError, Stage, check, resolve


class S(Stage):
    def __init__(self, needs=None, provides=None, inner=None):
        super().__init__(needs=needs, provides=provides)
        self._inner = inner or []

    def check_inner(self):
        return list(self._inner)


def chain():
    return {
        "a": S(needs={"raw": {"x"}}, provides={"p1": {"l1"}}),
        "b": S(needs={"p1": {"l1"}}, provides={"p2": {"l2"}}),
        "c": S(needs={"p2": {"l2"}}, provides={"p3": {"l3"}}),
        "extra": S(needs={"p1": {"l1"}}, provides={"side": {"z"}}),
    }


def test_resolve_is_backward_from_target():
    stages = chain()
    assert resolve(stages, "p3", {"raw": {"x"}}) == ["a", "b", "c"]
    assert resolve(stages, "p2", {"raw": {"x"}}) == ["a", "b"]


def test_stage_nobody_needs_is_not_resolved():
    stages = chain()
    assert "extra" not in resolve(stages, "p3", {"raw": {"x"}})
    assert "extra" in resolve(stages, "side", {"raw": {"x"}})


def test_missing_port_is_named():
    with pytest.raises(ContractError, match=r"a: needs port 'raw'"):
        resolve(chain(), "p3", {})


def test_ambiguous_provider_is_rejected():
    stages = chain()
    stages["dup"] = S(provides={"p2": {"l2"}})
    with pytest.raises(ContractError, match="several stages"):
        resolve(stages, "p3", {"raw": {"x"}})


def test_cycle_is_rejected():
    stages = {"a": S(needs={"q": set()}, provides={"p": set()}),
              "b": S(needs={"p": set()}, provides={"q": set()})}
    with pytest.raises(ContractError, match="cycle"):
        resolve(stages, "p", {})


def test_check_names_component_port_and_missing_label():
    stages = chain()
    stages["b"].needs = {"p1": {"l1", "l_missing"}}
    order = resolve(stages, "p3", {"raw": {"x"}})
    problems = check(stages, order, {"raw": {"x"}})
    assert len(problems) == 1
    p = problems[0]
    assert (p.component, p.port, p.missing) == ("b", "p1", ("l_missing",))
    assert "b: needs p1.['l_missing']" in str(p)


def test_check_ok_when_labels_match():
    stages = chain()
    order = resolve(stages, "p3", {"raw": {"x"}})
    assert check(stages, order, {"raw": {"x"}}) == []


def test_check_collects_inner_problems():
    from megobin.contracts import Problem

    stages = chain()
    stages["b"] = S(needs={"p1": {"l1"}}, provides={"p2": {"l2"}},
                    inner=[Problem("b/loss", "z", ("var",), ("mu",), "encoder E")])
    order = resolve(stages, "p3", {"raw": {"x"}})
    assert [p.component for p in check(stages, order, {"raw": {"x"}})] == ["b/loss"]
