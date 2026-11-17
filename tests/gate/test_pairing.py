import numpy as np

from skillrc.gate.pairing import ArmData, PairedTable
from skillrc.gate.schema import Episode
from skillrc.gate.spec import Grid

from conftest import arm, episodes, uid


def test_scored_attempt_is_first_without_infra_failure():
    eps = episodes("A", 10, lambda i, r: 1.0, infra=[(3, 1)])
    a = arm("A", eps, 10, max_infra_frac=0.2)          # 1/10 cells had an infra failure, cap 20%
    cell = (uid(3), 0, 1)
    assert a.scored[cell].attempt == 2
    assert cell in a.infra_cells and not a.invalid
    a2 = arm("A", eps, 10, max_infra_frac=0.05)        # 10% > 5% -> INVALID
    assert any("infrastructure" in r for r in a2.invalid)


def test_duplicate_records_and_double_scoring_are_invalid():
    eps = episodes("A", 5, lambda i, r: 1.0)
    dup = eps + [eps[0]]
    assert any("duplicate" in r for r in arm("A", dup, 5).invalid)
    second = eps[0].model_copy(update={"attempt": 2})
    assert any("more than one scored" in r for r in arm("A", eps + [second], 5).invalid)


def test_missing_cells_cap_and_cells_outside_grid():
    eps = episodes("A", 400, lambda i, r: 1.0, skip=[(1, 1)])
    a = arm("A", eps, 400)                       # 1/400 = 0.25% <= 0.5%
    assert not a.invalid and len(a.missing) == 1
    b = arm("A", episodes("A", 400, lambda i, r: 1.0, skip=[(1, 1), (2, 1), (3, 1)]), 400)
    assert any("missing" in r for r in b.invalid)
    c = arm("A", episodes("A", 12, lambda i, r: 1.0), 10)
    assert any("outside the declared grid" in r for r in c.invalid)


def test_task_identity_mismatch_is_invalid():
    # the ALFWorld re-download lesson: same task ids, different underlying tasks
    a = arm("A", episodes("A", 6, lambda i, r: 1.0), 6)
    shuffled = [e.model_copy(update={"task_uid": f"toy:{e.task_id}:{'f' * 16}"}) for e in episodes("B", 6, lambda i, r: 0.0)]
    b = arm("B", shuffled, 6)
    t = PairedTable(a, b)
    assert any("identity mismatch" in r for r in t.invalid)


def test_different_declared_task_sets_are_invalid():
    a = arm("A", episodes("A", 6, lambda i, r: 1.0), 6)
    b = arm("B", episodes("B", 5, lambda i, r: 1.0), 5)
    assert any("different task sets" in r for r in PairedTable(a, b).invalid)


def test_legacy_logs_are_descriptive():
    a = arm("A", episodes("A", 4, lambda i, r: 1.0, legacy=True), 4, grid=False)
    assert any("legacy" in r for r in a.descriptive)
    assert any("manifest" in r for r in a.descriptive)


def test_average_pairing_values():
    a = arm("A", episodes("A", 4, lambda i, r: 0.5 * r, reps=(1, 2)), 4, reps=(1, 2))
    b = arm("B", episodes("B", 4, lambda i, r: 0.25, reps=(1, 2)), 4, reps=(1, 2))
    t = PairedTable(a, b)
    assert np.allclose(t.diff("score"), 0.75 - 0.25)


def test_same_block_pairing_removes_task_by_block_interaction():
    # round-4 counterexample (c): no intervention effect, H_a scores 0 then 1 across blocks, H_b 1 then 0.
    n = 40
    half = lambda i: "a" if i % 2 == 0 else "b"            # noqa: E731
    truth = lambda i, k: (0.0 if k == 1 else 1.0) if half(i) == "a" else (1.0 if k == 1 else 0.0)  # noqa: E731
    abl = arm("ABL", episodes("ABL", n, lambda i, r: truth(i, r), reps=(1, 2), block=lambda i, r: r), n, reps=(1, 2))
    comp_block = lambda i: 1 if half(i) == "a" else 2      # noqa: E731
    comp = arm("COMP", episodes("COMP", n, lambda i, r: truth(i, comp_block(i)), block=lambda i, r: comp_block(i)), n)
    t = PairedTable(comp, abl)
    avg = t.diff("score", "average")
    same = t.diff("score", "same_block")
    assert np.allclose(same, 0.0)                         # correct: no effect
    assert np.allclose(np.abs(avg), 0.5)                  # averaging all repetitions is contaminated
