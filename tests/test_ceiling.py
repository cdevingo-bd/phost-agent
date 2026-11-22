import pytest

from phost_agent import Lease, Task, check_ceiling, gate_from_report, recorded_gate
from conftest import SCENE, STAGE


def report(share=0.5, ceiling="auto", applies=True, reason=None, enclosing=None, native_ms=1.0, worlds=(1,), stage=STAGE):
    if ceiling == "auto":
        ceiling = None if enclosing or share is None or share >= 1 else 1.0 / (1.0 - share)
    cells = [dict(worlds=w, nv=32, step_ms=2.0, stages={stage: dict(swap_class="exact", applies=applies, reason=reason, share=share,
                                                                  ceiling=ceiling, native_ms=native_ms, enclosing=enclosing)})
             for w in worlds]
    return dict(scene=SCENE, cells=cells)


def gate(rep, swap_class="exact", **kw):
    return gate_from_report(rep, SCENE, STAGE, swap_class=swap_class, **kw)


def test_a_stage_with_room_to_gain_passes_and_says_how_much():
    g = gate(report(share=0.75))
    assert g.ok and g.ceiling == pytest.approx(4.0) and "4.00x" in g.reason and g.bounded


def test_a_stage_that_is_a_sliver_of_the_step_is_refused_before_anything_is_proposed():
    g = gate(report(share=0.02))
    assert not g.ok and "1.020x" in g.reason and "2.0% of the step" in g.reason


def test_the_most_favourable_world_count_decides():
    rep = report(share=0.02)
    rep["cells"].append(report(share=0.6, worlds=(8,))["cells"][0])
    g = gate(rep)
    assert g.ok and g.worlds == 8


def test_a_stage_timed_only_inside_a_larger_phase_is_refused_on_its_bound():
    g = gate(report(share=0.03, enclosing="forward"))
    assert not g.ok and g.reason.startswith("at most 1.031x") and g.enclosing == "forward"
    assert gate(report(share=0.5, enclosing="forward")).ok


def test_a_stage_the_host_refuses_is_refused_with_the_hosts_reason():
    g = gate(report(applies=False, reason="no free joints"))
    assert not g.ok and "no free joints" in g.reason
    assert not gate_from_report(report(stage="other"), SCENE, STAGE, swap_class="exact").ok


def test_a_stage_with_no_native_timing_is_passed_with_its_ceiling_marked_unknown():
    g = gate(report(share=None, ceiling=None, native_ms=None))
    assert g.ok and not g.bounded and "unknown" in g.reason


@pytest.mark.parametrize("klass", ["model_changing", "observer"])
def test_a_stage_that_has_no_speed_to_gain_is_refused_for_a_speed_search(klass):
    g = gate(report(share=0.9), swap_class=klass)
    assert not g.ok and klass in g.reason and "behaviour" in g.reason


def test_the_behaviour_goal_passes_stages_with_no_ceiling():
    g = gate(report(share=0.9), swap_class="model_changing", goal="behaviour")
    assert g.ok and "no speed ceiling" in g.reason


# ---- measuring and remembering -------------------------------------------------------------------------

def test_the_scene_is_measured_once_and_then_remembered(journal):
    calls = []

    def measure(task, world):
        calls.append(task.scene)
        return report(share=0.5)

    task = Task(SCENE, STAGE, worlds=(1,))
    first, spent = check_ceiling(task, journal, measure=measure)
    second, again = check_ceiling(task, journal, measure=measure)
    assert first.basis == "measured" and second.basis == "memo" and len(calls) == 1 and again == 0.0 and spent >= 0.0
    assert first.ceiling == second.ceiling
    check_ceiling(task, journal, measure=measure, refresh=True)
    assert len(calls) == 2


def test_one_measurement_serves_every_stage_of_the_scene(journal):
    rep = report(share=0.5)
    rep["cells"][0]["stages"]["control.law"] = dict(swap_class="model_changing", applies=True, reason=None, share=0.1,
                                                    ceiling=None, native_ms=0.1, enclosing=None)
    calls = []
    measure = lambda t, w: calls.append(1) or rep
    check_ceiling(Task(SCENE, STAGE, worlds=(1,)), journal, measure=measure)
    g, _ = check_ceiling(Task(SCENE, "control.law", worlds=(1,), goal="behaviour"), journal, measure=measure)
    assert len(calls) == 1 and g.ok and g.basis == "memo"


def test_an_old_measurement_is_not_reused(journal):
    calls = []
    measure = lambda t, w: calls.append(1) or report(share=0.5)
    task = Task(SCENE, STAGE, worlds=(1,))
    check_ceiling(task, journal, measure=measure, now=lambda: 1000.0)
    check_ceiling(task, journal, measure=measure, now=lambda: 1000.0 + 3600, max_age_seconds=7200)
    check_ceiling(task, journal, measure=measure, now=lambda: 1000.0 + 9000, max_age_seconds=7200)
    assert len(calls) == 2


def test_a_measurement_that_failed_is_a_refusal_and_is_not_remembered(journal):
    task = Task(SCENE, STAGE, worlds=(1,))
    g, _ = check_ceiling(task, journal, measure=lambda t, w: dict(error="worker exited with code 1 and no result"))
    assert not g.ok and "could not be measured" in g.reason and journal.records("ceiling") == []


def test_the_device_is_leased_while_the_ceiling_is_measured(journal, tmp_path):
    lease = Lease("cpu", tmp_path)
    seen = []
    task = Task(SCENE, STAGE, worlds=(1,))
    check_ceiling(task, journal, measure=lambda t, w: seen.append(Lease("cpu", tmp_path).holder()) or report(), lease=lease)
    assert seen[0]["holder"].startswith("ceiling") and lease.holder() is None


def test_nothing_is_recorded_until_a_measurement_exists(journal):
    task = Task(SCENE, STAGE, worlds=(1,))
    assert recorded_gate(task, journal) is None
    check_ceiling(task, journal, measure=lambda t, w: report(share=0.5))
    assert recorded_gate(task, journal).ok


def test_a_scene_whose_engine_is_not_installed_is_refused_with_the_reason(journal, monkeypatch):
    import phost_agent.ceiling as module
    from phost_agent.task import World

    real = module.resolve

    def resolve(task, registry=None):
        world = real(task, registry)
        return World(world.registry, world.scene, world.engine_name, world.device_kind, "jax is not installed")

    monkeypatch.setattr(module, "resolve", resolve)
    g, spent = check_ceiling(Task(SCENE, STAGE), journal, measure=lambda t, w: pytest.fail("must not measure"))
    assert not g.ok and "jax is not installed" in g.reason and spent == 0.0
