"""Multi-basal coincidence cell: N source-distinct learned basal afferents under ONE gate.

The historical C owns exactly one basal. A direct-identity column gives its C one basal per
local ordinary E, so several local owners can each mature their own association. The
contract this file pins down:

    B = a current or one-boundary-carried basal event on ANY source
    A = any current apical event
    deposit at most once iff B AND A, using the CAUSAL source's own weight

and, critically, learning touches ONLY the causal weight -- there is no +1/-1 participation
term across the basal vector, so one owner's coincidence never depresses another's.
A one-basal cell must stay behaviorally identical (asserted here and by the goldens).
"""

import numpy as np
import pytest

from snn.neurons import CoincidencePyramidalNeuron, E_THRESHOLD

THETA = E_THRESHOLD
SRC = ['E0', 'E1', 'E2']
EDG = ['b0', 'b1', 'b2']


def make_multi(**kw):
    d = dict(nid='C', basal_source=list(SRC), basal_edge_id=list(EDG),
             apical_sources=['P0', 'P1'], apical_edge_ids=['a0', 'a1'],
             basal_weight=[100.0, 200.0, 300.0], w_max=8.0 * THETA, eta_c=0.01,
             threshold=THETA, leak_rate=0.0, refractory_steps=0)
    d.update(kw)
    return CoincidencePyramidalNeuron(**d)


def boundary(c, basal=(), apical=(), signal=1.0):
    """One full boundary: receipts, gate resolution, end-of-boundary eligibility settle."""
    c.begin_event_boundary()
    for s in basal:
        c.gather_basal(s, signal)
    for s in apical:
        c.gather_apical(s)
    q = c.resolve_dendrites()
    c.settle_eligibility()
    return q


# --------------------------------------------------------------- construction
def test_scalar_and_sequence_forms_agree_for_one_basal():
    a = CoincidencePyramidalNeuron('C', 'E0', 'b0', apical_sources=['P'],
                                   apical_edge_ids=['a'], basal_weight=250.0)
    b = CoincidencePyramidalNeuron('C', ['E0'], ['b0'], apical_sources=['P'],
                                   apical_edge_ids=['a'], basal_weight=[250.0])
    for cell in (a, b):
        assert cell.n_basal == 1
        assert cell.basal.source_ids == ['E0'] and cell.basal.edge_ids == ['b0']
        assert cell.basal_weight == 250.0
    assert np.array_equal(a.basal.weights, b.basal.weights)


def test_source_distinct_and_aligned():
    c = make_multi()
    assert c.n_basal == 3
    assert c.basal.source_ids == SRC and c.basal.edge_ids == EDG
    assert c.basal_weights_by_source() == {'E0': 100.0, 'E1': 200.0, 'E2': 300.0}
    assert c.basal.distance_factors.shape == (3,)


def test_duplicate_sources_rejected():
    with pytest.raises(ValueError, match='source-distinct'):
        make_multi(basal_source=['E0', 'E0', 'E1'])


def test_misaligned_weight_vector_rejected():
    with pytest.raises(ValueError, match='basal_weight must have 3 values'):
        make_multi(basal_weight=[1.0, 2.0])


def test_unknown_basal_source_rejected():
    c = make_multi()
    c.begin_event_boundary()
    with pytest.raises(ValueError, match='no basal afferent'):
        c.gather_basal('NOT_A_SOURCE')


# ------------------------------------------------- independent causal sources
@pytest.mark.parametrize('src,expect', [('E0', 100.0), ('E1', 200.0), ('E2', 300.0)])
def test_every_source_participates_with_its_own_weight(src, expect):
    c = make_multi()
    c.V = 0.0
    q = boundary(c, basal=[src], apical=['P0'])
    assert q == expect                              # the CAUSAL source selects the weight
    assert c.deposit_source == src
    assert c.coincidence_active and c.coincidence_deposit_count == 1


def test_learning_touches_only_the_causal_weight():
    c = make_multi(eta_c=0.5)
    before = c.basal_weights.copy()
    boundary(c, basal=['E1'], apical=['P0'])
    c.fire(0.5)
    c.update_basal_weight()
    after = c.basal_weights
    assert after[1] > before[1]                     # the causal association matured
    # every OTHER basal weight is byte-identical: no participation depression
    assert after[0] == before[0] and after[2] == before[2]


def test_two_owners_mature_independently():
    """The whole point of the multi-basal C: a second local owner learning its own
    association must not erase the first owner's."""
    c = make_multi(eta_c=0.5)
    for _ in range(20):
        boundary(c, basal=['E0'], apical=['P0'])
        c.fire(0.5)
        c.update_basal_weight()
    after_first = c.basal_weights.copy()
    assert after_first[0] > 100.0
    for _ in range(20):
        boundary(c, basal=['E2'], apical=['P0'])
        c.fire(0.5)
        c.update_basal_weight()
    assert c.basal_weights[2] > 300.0               # second owner matured
    assert c.basal_weights[0] == after_first[0]     # first owner untouched, exactly
    assert c.basal_weights[1] == 200.0              # never-participating source untouched


# -------------------------------------------------------------- the gate rule
def test_basal_only_and_apical_only_deposit_zero():
    for basal, apical in ((['E1'], ()), ((), ['P0'])):
        c = make_multi()
        assert boundary(c, basal=basal, apical=apical) == 0.0
        assert c.V == 0.0 and not c.coincidence_active


def test_sustained_basal_only_never_charges():
    c = make_multi()
    for _ in range(50):
        boundary(c, basal=['E2'])
    assert c.V == 0.0 and c.coincidence_deposit_count == 0


@pytest.mark.parametrize('src,expect', [('E0', 100.0), ('E1', 200.0), ('E2', 300.0)])
def test_one_boundary_carry_works_per_source(src, expect):
    c = make_multi()
    assert boundary(c, basal=[src]) == 0.0          # basal alone: carried, not deposited
    assert c.basal_eligible
    assert boundary(c, apical=['P1']) == expect     # apical next boundary consumes the carry
    assert c.deposit_source == src


def test_carry_expires_after_exactly_one_boundary():
    c = make_multi()
    boundary(c, basal=['E2'])
    boundary(c)                                     # a silent boundary expires the carry
    assert not c.basal_eligible
    assert boundary(c, apical=['P0']) == 0.0


def test_consumed_carry_cannot_be_reused():
    c = make_multi()
    boundary(c, basal=['E1'])
    assert boundary(c, apical=['P0']) == 200.0      # consumed here
    assert boundary(c, apical=['P0']) == 0.0        # and not again


def test_current_event_is_preferred_over_a_carried_one():
    c = make_multi()
    boundary(c, basal=['E0'])                       # E0 carried
    q = boundary(c, basal=['E2'], apical=['P0'])    # E2 arrives now
    assert q == 300.0 and c.deposit_source == 'E2'  # current wins


# ------------------------------------------ deterministic multi/duplicate rule
def test_multiple_distinct_sources_earliest_delivered_is_causal():
    c = make_multi()
    c.begin_event_boundary()
    c.gather_basal('E2')                            # arrives first
    c.gather_basal('E0')
    c.gather_apical('P0')
    q = c.resolve_dendrites()
    assert q == 300.0 and c.deposit_source == 'E2'  # arrival order decides, not index order
    assert c.basal_extra_sources == ['E0']          # the second is RECORDED, not discarded
    assert c.basal_sources == ['E2', 'E0']


def test_duplicate_receipt_on_one_source_is_counted_not_re_deposited():
    c = make_multi()
    c.begin_event_boundary()
    c.gather_basal('E1')
    c.gather_basal('E1')                            # duplicate routing
    c.gather_apical('P0')
    q = c.resolve_dendrites()
    assert q == 200.0
    assert c.basal_duplicate_count == 1 and c.basal_delivery_count == 2
    assert c.basal_extra_sources == []              # a duplicate is not a second source


def test_one_coincidence_deposits_at_most_once():
    c = make_multi()
    c.begin_event_boundary()
    c.gather_basal('E0')
    c.gather_apical('P0')
    first = c.resolve_dendrites()
    again = c.resolve_dendrites()                   # idempotent for the rest of the boundary
    c.gather_apical('P1')
    third = c.resolve_dendrites(tau=0.9)
    assert (first, again, third) == (100.0, 0.0, 0.0)
    assert c.coincidence_deposit_count == 1 and c.V == 100.0


def test_deposit_and_firing_share_the_permitting_apical_tau():
    c = make_multi(basal_weight=[THETA, THETA, THETA])
    c.begin_event_boundary()
    c.gather_basal('E1')
    c.deliver_apical('P0', 0.42)                    # permission arrives at its own tau
    assert c.coincidence_deposit_tau == 0.42
    assert c.can_fire()
    c.fire(0.42)
    assert c.spike_tau == 0.42


def test_retained_voltage_cannot_fire_without_a_current_gate():
    c = make_multi(basal_weight=[THETA, THETA, THETA])
    boundary(c, basal=['E0'], apical=['P0'])        # charges to threshold
    assert c.V >= c.threshold
    c.begin_event_boundary()                        # new boundary, no receipts
    assert not c.coincidence_active and not c.can_fire()
    assert c.crossing_time(1.0) == float('inf')


# --------------------------------------------------------------- the theta cap
def test_each_basal_weight_is_capped_independently_at_theta():
    c = make_multi(basal_weight=[0.9 * THETA] * 3, w_cap=THETA, eta_c=5.0,
                   update_mode='c_dual_fe_fes')
    for src in ('E0', 'E2'):
        for _ in range(600):
            boundary(c, basal=[src], apical=['P0'])
            c.v_pre = THETA
            c.fire(0.5)
            c.update_basal_weight()
    assert c.basal_weights[0] == THETA and c.basal_weights[2] == THETA
    assert c.basal_weights[1] == 0.9 * THETA        # untrained source unmoved
