import pytest

from phost_agent import Budget, Spend, exhausted


def test_a_fresh_spend_is_within_the_default_budget():
    assert exhausted(Budget(), Spend()) is None


@pytest.mark.parametrize("spend,budget,text", [
    (Spend(attempts=3), Budget(attempts=3), "attempt budget"),
    (Spend(gpu_seconds=10.0), Budget(gpu_seconds=10.0), "gpu-seconds"),
    (Spend(cpu_seconds=5.0), Budget(cpu_seconds=5.0), "cpu-seconds"),
    (Spend(tokens=100), Budget(tokens=100), "token budget"),
    (Spend(stalled=2), Budget(stall=2), "stalled"),
])
def test_each_limit_says_which_it_was(spend, budget, text):
    assert text in exhausted(budget, spend)


def test_unset_limits_do_not_stop_a_loop():
    assert exhausted(Budget(cpu_seconds=None, tokens=None), Spend(cpu_seconds=1e9, tokens=10 ** 9)) is None


def test_more_workers_need_a_cpu_budget_so_a_wider_search_is_always_budgeted():
    assert Budget().workers == 1
    with pytest.raises(ValueError, match="cpu_seconds"):
        Budget(workers=4)
    assert Budget(workers=4, cpu_seconds=60.0).workers == 4


@pytest.mark.parametrize("bad", [dict(attempts=0), dict(stall=0), dict(workers=0), dict(gpu_seconds=-1.0)])
def test_nonsense_budgets_are_refused(bad):
    with pytest.raises(ValueError):
        Budget(**bad)
