"""Delay-1 top-down feedback hard reset (tiled cortical columns).

Covers the ``c_feedback_reset`` mechanism: a firing column C schedules, via the same
column ``I``, a delay-1 hard reset of the column's ordinary-E bank applied at the START
of the next boundary (after ``freeze_drive``, before the event loop). This suppresses a
trained instant integrator's redundant re-fire and drops the confirmed column toward the
frequency-halving cadence. The zero-latency WTA ``E->I`` reset is untouched, and this path
never uses persistent inhibitory conductance (``g_inh`` stays 0).

The mechanics tests inject a deterministic "trained instant integrator" via
``stimulate(..., continuous=True)`` and drive the feedback queue directly, so they do not
depend on the multi-thousand-step training the real C-firing path needs. The integration
test trains the real hierarchy until C fires and measures the on/off output-rate delta.
"""

from backend.simulation import SimulationEngine, CoincidencePyramidalNeuron


THETA_MAG = 1.2   # continuous injection > theta -> fires every boundary (instant integrator)


def _one_column(eng):
    """Return (column_id, i_id, [ordinary E ids]) for an arbitrary tiled column."""
    for cid in sorted({c for c in eng._column_of.values() if c}):
        e_ids = sorted(nid for nid, c in eng._column_of.items()
                       if c == cid and eng._role_of.get(nid) == 'E')
        i_ids = [nid for nid, c in eng._column_of.items()
                 if c == cid and eng._role_of.get(nid) == 'I']
        if e_ids and i_ids:
            return cid, i_ids[0], e_ids
    raise AssertionError('no tiled column found')


def _schedule_feedback_like_c(eng, i_id):
    """Queue the column I's hard-reset targets for the next boundary, exactly as a firing
    C's column_c_to_i trigger does. Appends to _hardreset_next (snapshotted at the top of
    the next step)."""
    for tgt, hr_eid in eng._hardreset_out.get(i_id, []):
        eng._hardreset_next.append((tgt, i_id, hr_eid))


# --------------------------------------------------------------------------- mechanics


def test_feedback_queue_drains_at_next_boundary_start():
    """A pending feedback reset wipes the whole column E bank at the next boundary start,
    records feedback_hard_reset events, and is consumed (single boundary)."""
    eng = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=True)
    cid, i_id, e_ids = _one_column(eng)
    target = e_ids[0]
    eng.stimulate(target, magnitude=THETA_MAG, continuous=True)

    # Warm: the instant integrator fires every boundary.
    for _ in range(3):
        eng.step()
    assert eng._spike_hist[target][-1] == 1, 'instant integrator should fire every boundary'

    # Schedule a feedback reset (as a C spike at t-1 would) and step.
    _schedule_feedback_like_c(eng, i_id)
    st = eng.step()

    assert eng._spike_hist[target][-1] == 0, 'pending feedback reset must suppress the fire'
    fb = [h for h in st['hard_reset_events'] if h['kind'] == 'feedback_hard_reset']
    assert {h['target'] for h in fb} == set(e_ids), 'the entire ordinary-E bank is reset'
    assert all(h['source'] == i_id and h['tau'] == 0.0 for h in fb)
    assert eng._hardreset_next == [], 'queue consumed; reset does not survive two boundaries'

    # No new pending reset -> the integrator fires again next boundary (one-shot).
    eng.step()
    assert eng._spike_hist[target][-1] == 1, 'suppression is exactly one boundary'


def test_feedback_reset_discards_frozen_drive_defeats_tau0_crosser():
    """The reset discards V and the just-frozen drive, so a trained tau~=0 crosser cannot
    fire -- proving a boundary-start reset (not a same-boundary one) is what suppresses it."""
    eng = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=True)
    cid, i_id, e_ids = _one_column(eng)
    target = e_ids[0]
    eng.stimulate(target, magnitude=THETA_MAG, continuous=True)
    for _ in range(3):
        eng.step()

    _schedule_feedback_like_c(eng, i_id)
    eng.step()
    cell = eng.exc[target]
    # After the boundary-start drain the cell was wiped and its supra-theta drive discarded.
    assert cell.V == cell.v_rest
    assert cell.remaining_excitation == 0.0
    assert not cell.spiked


def test_same_boundary_reset_would_be_noop_on_the_winner():
    """The reason the feedback reset is DEFERRED: a trained winner crosses at tau~=0 and
    fire() already set V=v_rest, so a same-boundary hard reset (at C's later tau) lands on
    an already-fired cell and changes nothing. This is why only a next-boundary reset can
    suppress the re-fire."""
    eng = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=True)
    cid, i_id, e_ids = _one_column(eng)
    target = e_ids[0]
    eng.stimulate(target, magnitude=THETA_MAG, continuous=True)
    for _ in range(3):
        eng.step()
    cell = eng.exc[target]
    assert eng._spike_hist[target][-1] == 1 and cell.V == cell.v_rest and cell.spiked
    # A same-boundary reset after the spike is inert: V is already rest, the spike stands.
    v_after = cell.hard_reset(0.5)
    assert v_after == cell.v_rest
    assert cell.spiked, 'the already-emitted spike is not undone by a later same-boundary reset'


def test_wta_selection_unchanged_by_gate_on_no_feedback_boundary():
    """On a boundary with no pending feedback, WTA is identical with the gate on or off:
    stimulating two competitors yields the same single winner and the same loser reset."""
    def winners(fb):
        eng = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=fb)
        cid, i_id, e_ids = _one_column(eng)
        eng.stimulate(e_ids[0], magnitude=THETA_MAG, continuous=True)
        eng.stimulate(e_ids[1], magnitude=1.1, continuous=True)
        seq = []
        for _ in range(10):
            st = eng.step()
            w = st['column_winners'].get(cid)
            seq.append(w['id'] if w else None)
        return seq
    assert winners(True) == winners(False), 'WTA winner selection must not depend on the gate'


def test_gate_off_never_schedules_and_is_byte_identical():
    """With c_feedback_reset=False the C->I trigger is the legacy guarded no-op: the queue
    is never populated and the tiled dynamics match the gate-on schedule that never fires
    a C (proven separately by the golden test)."""
    eng_off = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=False)
    cid, i_id, e_ids = _one_column(eng_off)
    eng_off.stimulate(e_ids[0], magnitude=THETA_MAG, continuous=True)
    for _ in range(20):
        eng_off.step()
        assert eng_off._hardreset_next == [], 'gate off must never schedule a feedback reset'


def test_wta_and_feedback_resets_coexist_on_one_bank():
    """A boundary that both applies a pending feedback reset AND runs the WTA E->I reset is
    consistent: idempotent wipe, no g_inh, exactly one ordinary-E winner selection logic."""
    eng = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=True)
    cid, i_id, e_ids = _one_column(eng)
    # Stimulate TWO E's: one will win WTA (drives I -> resets the other). A pending feedback
    # reset also lands on the whole bank at boundary start.
    eng.stimulate(e_ids[0], magnitude=THETA_MAG, continuous=True)
    eng.stimulate(e_ids[1], magnitude=THETA_MAG, continuous=True)
    for _ in range(3):
        eng.step()
    _schedule_feedback_like_c(eng, i_id)
    st = eng.step()   # must not raise
    # Feedback wiped the bank at boundary start -> no ordinary E crosses -> no winner here.
    assert cid not in st['column_winners']
    assert all(eng.exc[e].g_inh == 0.0 for e in e_ids), 'no persistent conductance on this path'


def test_ginh_stays_zero_with_feedback_on():
    """The whole point of this topology is hard-reset-only: g_inh must remain identically 0
    for every excitatory/coincidence cell across a feedback-active run."""
    eng = SimulationEngine(seed=1, topology='tiled_cc', c_feedback_reset=True)
    cid, i_id, e_ids = _one_column(eng)
    eng.stimulate(e_ids[0], magnitude=THETA_MAG, continuous=True)
    gmax = 0.0
    for _ in range(50):
        if eng.timestep % 5 == 0:
            _schedule_feedback_like_c(eng, i_id)
        eng.step()
        gmax = max(gmax, max(abs(n.g_inh) for n in eng.exc.values()))
    assert gmax == 0.0


# ------------------------------------------------------------------------- integration


def _train_and_measure(fb, warm=4000, measure=1000, seed=1):
    # Measured at the dashboard's FAST maturation rates. At the engine-default eta=0.01 the
    # top-down loop is still immature after a short warm-up, so the on/off gap is a couple of
    # boundaries out of ~540 -- too small to be evidence of anything. These rates mature both
    # the L2 feedforward and the coincidence C, where the reduction is a robust ~3%.
    eng = SimulationEngine(seed=seed, topology='tiled_cc', c_feedback_reset=fb,
                           dual_fe_fes=True, eta=4.0, c_eta=2.0, dual_fe_B=5.0,
                           leak_rate=0.0, refractory_steps=0, e_weight_cap_frac=0.5)
    c_ids = [n.id for n in eng.exc.values() if isinstance(n, CoincidencePyramidalNeuron)]
    eng.set_pattern('row 1')
    for _ in range(warm):
        eng.step()
    fb_events = 0
    c_spikes = {}
    e_winners = {}
    gmax = 0.0
    for _ in range(measure):
        st = eng.step()
        for h in st['hard_reset_events']:
            if h['kind'] == 'feedback_hard_reset':
                fb_events += 1
        for c in c_ids:
            if eng.spiked.get(c):
                c_spikes[c] = c_spikes.get(c, 0) + 1
        for col in st['column_winners']:
            e_winners[col] = e_winners.get(col, 0) + 1
        gmax = max(gmax, max(abs(n.g_inh) for n in eng.exc.values()))
    return dict(fb_events=fb_events, c_spikes=c_spikes, e_winners=e_winners, gmax=gmax)


def test_halving_confirmed_column_fires_strictly_less():
    """Trained tiled_cc: the confirmed column (its C fires) fires strictly less often with
    feedback on than off; feedback events occur only when on; g_inh stays 0. The exact
    ratio is loop-latency determined (the loop re-times as spikes are suppressed), so we
    assert a strict reduction and report the measured ratio rather than hard-coding 1/2."""
    on = _train_and_measure(True)
    off = _train_and_measure(False)

    assert on['fb_events'] > 0, 'feedback path must activate once C fires'
    assert off['fb_events'] == 0, 'feedback path silent when gated off'
    assert on['gmax'] == 0.0 and off['gmax'] == 0.0, 'hard-reset only; no g_inh'
    assert on['c_spikes'], 'the trained hierarchy must fire at least one C cell'

    confirmed_c = max(on['c_spikes'], key=on['c_spikes'].get)
    confirmed_col = confirmed_c[:-1]   # strip trailing 'C' -> column id
    on_rate = on['e_winners'].get(confirmed_col, 0)
    off_rate = off['e_winners'].get(confirmed_col, 0)
    ratio = on_rate / off_rate if off_rate else float('nan')
    print(f'\nconfirmed column {confirmed_col}: E-winner boundaries on={on_rate} '
          f'off={off_rate} ratio={ratio:.3f} (feedback events on={on["fb_events"]})')

    assert off_rate > 0
    assert on_rate < off_rate, 'confirmed column must fire strictly less often with feedback on'

    # A column with no ordinary-E activity at all (no input, no feedback) is untouched by
    # the feature: absent from both on and off winner maps.
    assert set(on['e_winners']) == set(off['e_winners']), (
        'feedback must not create or remove activity in columns it does not touch')
