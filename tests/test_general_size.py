"""Input/output-count generality: the engine must build and run at grid sizes other
than the default 9/8, with correctly sized layout, distance, and feedforward arrays.

The 9/8 goldens (tests/golden/*) pin bit-exactness at the default size; these tests
pin that nothing latently fixes the shape to a 3-wide grid.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT
from backend.layout import grid_dims, generate_layout


def test_grid_dims_are_near_square():
    assert grid_dims(9) == (3, 3)             # the historical default sheet
    assert grid_dims(16) == (4, 4)            # perfect square
    assert grid_dims(12) == (4, 3)            # width grows first, last row partial
    assert grid_dims(1) == (1, 1)


def test_defaults_unchanged():
    e = SimulationEngine(seed=1)              # no size overrides
    assert (e.n_pix, e.n_out) == (N_PIX, N_OUT) == (9, 8)


@pytest.mark.parametrize('topology', ['pi', 'old', 'rg'])
def test_engine_builds_and_steps_at_non_default_size(topology):
    n_pix, n_out = 16, 12
    e = SimulationEngine(seed=3, topology=topology, n_pix=n_pix, n_out=n_out)
    assert (e.n_pix, e.n_out) == (n_pix, n_out)

    # Competitor population and its plastic afferent bank are sized to the real counts.
    assert len(e.l2e) == n_out
    for c in e.l2e:
        assert c.acc_weights.shape == (n_pix,)                 # dense fan-in == n_pix
        assert c.acc_distance_factor.shape == (n_pix,)         # aligned distance factors
        # 1/d^2 factors are real geometry, not a placeholder of ones.
        assert np.all(c.acc_distance_factor > 0.0)
        assert not np.allclose(c.acc_distance_factor, 1.0)

    # A random binary pattern of the RIGHT length drives the network with no shape error;
    # the weight rule runs over the actual afferent count as we step.
    rng = np.random.default_rng(0)
    e.set_input((rng.random(n_pix) > 0.5).astype(float))
    assert e.input_vec.shape == (n_pix,)
    before = np.stack([c.acc_weights.copy() for c in e.l2e])
    for _ in range(30):
        e.step()
    after = np.stack([c.acc_weights.copy() for c in e.l2e])
    assert after.shape == before.shape == (n_out, n_pix)
    # Where competitors have direct sensory afferents (pi/old), the accumulating rule
    # runs over the actual n_pix-wide bank -- confirm it moved weights. (rg learns L2E
    # through a two-hop encoder path that need not engage under a single static input.)
    if topology in ('pi', 'old'):
        assert not np.array_equal(after, before)              # learning ran over n_pix afferents

    # Reported display grid tracks the actual sheet dimensions.
    cols, rows = grid_dims(n_pix)
    assert e.topology()['grid'] == dict(rows=rows, cols=cols)


def test_layout_positions_are_generic_and_distinct():
    # A non-square count still lays every sensory cell on a distinct grid anchor.
    rng = np.random.default_rng(1)
    n_pix, n_out = 12, 6
    pos = generate_layout(rng, n_pix, n_out)
    anchors = {tuple(np.round(pos[f'L1E{i}'], 3)) for i in range(n_pix)}
    assert len(anchors) == n_pix                              # no two pixels collide
    assert all(f'L2E{j}' in pos for j in range(n_out))


@pytest.mark.parametrize('topology', ['rg_residual', 'rg_coincidence'])
def test_event_resolved_topologies_boot_blank_at_non_default_size(topology):
    # Event-resolved topologies read self.input_vec[pixel] for every RG pixel. The engine
    # must NOT seed a 9-length demo pattern into a differently sized input (regression:
    # that raised IndexError on the first step).
    n_pix, n_out = 16, 12
    e = SimulationEngine(seed=5, topology=topology, n_pix=n_pix, n_out=n_out)
    assert e.current_pattern is None                         # no ill-fitting demo pattern
    assert e.input_vec.shape == (n_pix,)
    assert e.input_vec.sum() == 0.0                          # blank until the caller fills it
    rng = np.random.default_rng(7)
    e.set_input((rng.random(n_pix) > 0.5).astype(float))
    for _ in range(20):
        e.step()                                            # no shape error over the RG sheet


def test_builtin_patterns_rejected_at_non_default_size():
    # The 3x3 demo patterns are 9-pixel; asking for one on a wider sheet fails loudly
    # rather than silently installing a mismatched input vector.
    e = SimulationEngine(seed=1, topology='pi', n_pix=16, n_out=8)
    with pytest.raises(ValueError):
        e.set_pattern('row 1')


def test_pattern_size_is_not_assumed_three():
    # A presented pattern may activate ANY subset of the inputs, not a fixed count.
    e = SimulationEngine(seed=2, topology='pi', n_pix=16, n_out=10)
    for active_count in (0, 1, 5, 16):
        vec = np.zeros(16)
        vec[:active_count] = 1.0
        e.set_input(vec)
        e.step()                                              # no assumption on the active count
        assert e.input_vec.sum() == active_count
