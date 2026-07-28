"""Two-tower V/A/7 composition experiment (Task 4): structure, glyphs, milestones.

Covers the focused tests declared for Task 4:

- explicit live-contract engine construction (dual FE/FES, eta/c_eta, theta/2 detector
  ceiling, fixed non-plastic Eor at theta, auto pacing, feedback reset);
- two-tower topology counts and COMPLETE connectivity (not just totals);
- tower isolation, parent metadata, and the dormant L3 C;
- exact population selection for E / Eor / C / I at L1, L2 and L3;
- glyph coordinates, half membership, distinctness and ASCII rendering;
- structural one-event maturity and observed-throughput bookkeeping from synthetic traces;
- representation-separability classification;
- failure taxonomy, replay slugs and resume completion markers.

No 30k-boundary scientific run lives here; the engine-level tests step a few hundred
boundaries at most.
"""

import json
import os

import pytest

from backend.network_spec import (
    TILED_FAMILY, SpecError, two_tower_composition_spec, validate_spec,
)
from experiments.two_tower_analysis import (
    CANONICAL_GLYPH_ORDER, GLYPH_COLS, GLYPH_ROWS, GLYPH_SEAM, FAILURE_TAXONOMY,
    L3Signature, SlugRegistry, ThroughputTracker, active_charge, c_readiness,
    classify_signatures, condition_slug, failure_record, glyph_ascii, glyph_coordinates,
    glyph_half_counts, glyph_patches, glyph_vector, one_event_maturity, verify_glyphs,
)
from experiments.two_tower_composition import (
    ENGINE_CONTRACT, FabricIndex, REFERENCE_B, assert_engine_contract, cell_is_complete,
    make_two_tower_engine, next_attempt_dir, topology_fingerprint, write_cell_status,
)

N_IN = GLYPH_ROWS * GLYPH_COLS      # 162


# ============================================================ topology structure
@pytest.fixture(scope="module")
def spec():
    return two_tower_composition_spec(cc_e_count=8)


def test_default_counts_are_393_nodes_2162_edges(spec):
    assert (len(spec["nodes"]), len(spec["edges"])) == (393, 2162)
    norm = validate_spec(spec, N_IN)
    assert (len(norm["nodes"]), len(norm["edges"])) == (393, 2162)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 12])
def test_arbitrary_n_follows_the_formulae(n):
    s = two_tower_composition_spec(cc_e_count=n)
    n_cols = 21
    assert len(s["nodes"]) == 162 + n_cols * (n + 3)
    assert len(s["edges"]) == n_cols * (3 * n + 2) + 18 * 9 * n + 20 * 2 * n
    validate_spec(s, N_IN)


def test_ids_unique_and_construction_deterministic_and_fresh():
    a, b = two_tower_composition_spec(), two_tower_composition_spec()
    ids = [n["id"] for n in a["nodes"]]
    eids = [e["id"] for e in a["edges"]]
    assert len(ids) == len(set(ids)) and len(eids) == len(set(eids))
    assert ids == [n["id"] for n in b["nodes"]]
    assert eids == [e["id"] for e in b["edges"]]
    a["nodes"].append({"poison": True})                 # fresh objects, no shared state
    assert len(b["nodes"]) == 393


def test_exactly_162_unique_rgc_pixels_on_one_9x18_sheet(spec):
    rgc = [n for n in spec["nodes"] if n["archetype"] == "rg_source"]
    assert len(rgc) == 162
    assert sorted(n["pixel"] for n in rgc) == list(range(162))
    for n in rgc:
        assert n["pixel"] == n["input_row"] * GLYPH_COLS + n["input_col"]
        assert (n["patch_row"], n["patch_col"]) == (n["input_row"] // 3, n["input_col"] // 3)
        assert n["patch_id"] == n["patch_row"] * 6 + n["patch_col"]
    # disjoint pixel ownership across the two 9x9 fields
    left = {n["pixel"] for n in rgc if n["input_col"] < GLYPH_SEAM}
    right = {n["pixel"] for n in rgc if n["input_col"] >= GLYPH_SEAM}
    assert len(left) == len(right) == 81 and not (left & right)


def test_column_census_and_composition(spec):
    meta = spec["topology"]
    assert meta["family"] == TILED_FAMILY
    assert meta["input_shape"] == dict(rows=9, cols=18)
    assert meta["grid_shape"] == dict(rows=3, cols=6)
    assert [L["layer"] for L in meta["column_layers"]] == ["L1", "L2", "L3"]
    by_layer = {}
    for c in meta["columns"]:
        by_layer.setdefault(c["layer"], []).append(c["id"])
    assert len(by_layer["L1"]) == 18 and len(by_layer["L2"]) == 2 and len(by_layer["L3"]) == 1
    roles = {c["id"]: {"E": 0, "Eor": 0, "C": 0, "I": 0} for c in meta["columns"]}
    for n in spec["nodes"]:
        if n.get("column_role"):
            roles[n["column_id"]][n["column_role"]] += 1
    for cid, r in roles.items():
        assert r == {"E": 8, "Eor": 1, "C": 1, "I": 1}, (cid, r)


def test_parent_metadata_and_dormant_c(spec):
    cols = {c["id"]: c for c in spec["topology"]["columns"]}
    has_parent = {n["column_id"]: n["has_parent"] for n in spec["nodes"]
                  if n.get("column_role") == "C"}
    for cid, c in cols.items():
        if c["layer"] == "L1":
            assert len(c["parent_ids"]) == 1 and cols[c["parent_ids"][0]]["layer"] == "L2"
        elif c["layer"] == "L2":
            assert c["parent_ids"] == ["L3c00"]
        else:
            assert c["parent_ids"] == []
        # C dormancy must AGREE with the declared parent relation
        assert has_parent[cid] == bool(c["parent_ids"])
    assert has_parent["L3c00"] is False


def test_every_rgc_feeds_only_its_own_patch_e_bank(spec):
    node = {n["id"]: n for n in spec["nodes"]}
    cols = {c["id"]: c for c in spec["topology"]["columns"]}
    seen = {}
    for e in spec["edges"]:
        if e["kind"] != "feedforward" or node[e["source"]]["archetype"] != "rg_source":
            continue
        src, tgt = node[e["source"]], node[e["target"]]
        assert tgt["column_role"] == "E"
        c = cols[tgt["column_id"]]
        assert (src["patch_row"], src["patch_col"]) == (c["row"], c["col"])
        seen.setdefault(tgt["id"], set()).add(src["id"])
    # every ordinary L1 E receives ALL nine RGCs of its patch, and nothing else
    l1_e = [n["id"] for n in spec["nodes"]
            if n.get("column_role") == "E" and cols[n["column_id"]]["layer"] == "L1"]
    assert len(l1_e) == 18 * 8
    assert all(len(seen[e]) == 9 for e in l1_e)
    non_l1_e = [n["id"] for n in spec["nodes"]
                if n.get("column_role") == "E" and cols[n["column_id"]]["layer"] != "L1"]
    assert all(e not in seen for e in non_l1_e)


def test_tower_isolation_and_l2_to_l3_links(spec):
    node = {n["id"]: n for n in spec["nodes"]}
    cols = {c["id"]: c for c in spec["topology"]["columns"]}
    l1_to_l2, l2_to_l3 = {}, {}
    for e in spec["edges"]:
        if e["kind"] != "feedforward":
            continue
        s, t = node[e["source"]], node[e["target"]]
        if s.get("column_role") != "Eor" or t.get("column_role") != "E":
            continue
        s_layer = cols[s["column_id"]]["layer"]
        t_layer = cols[t["column_id"]]["layer"]
        if s_layer == "L1":
            assert t_layer == "L2"
            l1_to_l2.setdefault(s["column_id"], set()).add(t["column_id"])
        else:
            assert (s_layer, t_layer) == ("L2", "L3")
            l2_to_l3.setdefault(s["column_id"], set()).add(t["column_id"])
    # every L1 links to exactly its OWN tower's L2, and to nothing else
    for cid, parents in l1_to_l2.items():
        assert parents == set(cols[cid]["parent_ids"])
        assert len(parents) == 1
    assert len(l1_to_l2) == 18
    # both and ONLY the two L2 columns connect to L3
    assert set(l2_to_l3) == {"T0L2c00", "T1L2c00"}
    assert all(v == {"L3c00"} for v in l2_to_l3.values())
    # no cross-tower L1 -> L2 edge exists at all
    towers = {"T0L2c00": set(), "T1L2c00": set()}
    for cid in l1_to_l2:
        towers[cols[cid]["parent_ids"][0]].add(cid)
    assert len(towers["T0L2c00"]) == len(towers["T1L2c00"]) == 9
    assert not (towers["T0L2c00"] & towers["T1L2c00"])


def test_apical_feedback_is_exactly_the_declared_parent(spec):
    node = {n["id"]: n for n in spec["nodes"]}
    cols = {c["id"]: c for c in spec["topology"]["columns"]}
    apical = {}
    for e in spec["edges"]:
        if e["kind"] != "apical_excitation":
            continue
        s, t = node[e["source"]], node[e["target"]]
        assert s["column_role"] == "E" and t["column_role"] == "C"
        apical.setdefault(t["column_id"], set()).add(s["column_id"])
    for cid, c in cols.items():
        if c["parent_ids"]:
            assert apical[cid] == set(c["parent_ids"])
        else:
            assert cid not in apical            # L3 C has ZERO apical inputs
    assert "L3c00" not in apical


def test_i_resets_stay_column_local(spec):
    node = {n["id"]: n for n in spec["nodes"]}
    n_reset = 0
    for e in spec["edges"]:
        if e["kind"] != "hard_reset_inhibition":
            continue
        s, t = node[e["source"]], node[e["target"]]
        assert s["column_role"] == "I" and t["column_role"] == "E"
        assert s["column_id"] == t["column_id"]
        n_reset += 1
    assert n_reset == 21 * 8


def test_no_feature_relay_or_direct_identity_variant(spec):
    assert "variant" not in spec["topology"]           # classic column motif only
    archetypes = {n["archetype"] for n in spec["nodes"]}
    assert archetypes == {"rg_source", "e_latency_competitor", "e_coincidence", "i_relay"}
    kinds = {e["kind"] for e in spec["edges"]}
    assert kinds == {"feedforward", "relay_excitation", "hard_reset_inhibition",
                     "basal_excitation", "apical_excitation"}
    projections = {e.get("projection") for e in spec["edges"]}
    assert projections == {"column_e_to_eor", "column_e_to_i", "column_i_to_e",
                           "column_eor_to_c_basal", "column_c_to_i", "rg_to_column",
                           "column_to_column_ff", "column_to_column_apical"}


def test_malformed_two_tower_graph_is_rejected(spec):
    import copy
    bad = copy.deepcopy(spec)
    # a cross-tower L1 -> L2 link must not validate
    for e in bad["edges"]:
        if e["source"] == "T0L1c00Eor" and e["target"] == "T0L2c00E0":
            e["target"] = "T1L2c00E0"
            break
    with pytest.raises(SpecError):
        validate_spec(bad, N_IN)


def test_builder_rejects_zero_e_count():
    with pytest.raises(ValueError):
        two_tower_composition_spec(cc_e_count=0)


# ============================================================ glyphs
def test_glyph_invariants_hold():
    report = verify_glyphs(CANONICAL_GLYPH_ORDER)
    assert set(report) == set(CANONICAL_GLYPH_ORDER)
    for g, rep in report.items():
        assert rep["n_active"] == len(rep["coordinates"])
        assert rep["half_counts"]["left"] > 0 and rep["half_counts"]["right"] > 0


def test_exact_glyph_coordinates():
    assert glyph_coordinates("V") == sorted(
        [(r, r) for r in range(9)] + [(r, 17 - r) for r in range(9)])
    assert glyph_coordinates("A") == sorted(set(
        [(r, 8 - r) for r in range(9)] + [(r, 9 + r) for r in range(9)]
        + [(4, c) for c in range(4, 14)]))
    assert glyph_coordinates("7") == sorted(set(
        [(0, c) for c in range(18)] + [(r, 17) for r in range(9)]))
    assert len(glyph_coordinates("V")) == 18
    assert len(glyph_coordinates("A")) == 26
    assert len(glyph_coordinates("7")) == 26


def test_glyph_vectors_are_distinct_and_correctly_sized():
    vecs = {g: glyph_vector(g) for g in CANONICAL_GLYPH_ORDER}
    assert all(len(v) == 162 for v in vecs.values())
    keys = {tuple(v) for v in vecs.values()}
    assert len(keys) == 3
    # V and A share diagonal structure but differ by the crossbar and the apex direction
    assert vecs["V"] != vecs["A"]


def test_glyph_ascii_matches_the_binary_vector():
    for g in CANONICAL_GLYPH_ORDER:
        vec = glyph_vector(g)
        art = glyph_ascii(g)
        assert len(art) == 9
        for r, line in enumerate(art):
            assert line[GLYPH_SEAM] == "|"
            cells = line.replace("|", "")
            for c, ch in enumerate(cells):
                assert (ch == "#") == bool(vec[r * 18 + c])


def test_glyph_halves_and_patch_activation():
    assert glyph_half_counts("V") == {"left": 9, "right": 9}
    # every active patch of every glyph has >= 2 active pixels, so a theta/2-capped
    # detector can still be driven over threshold by its own patch
    for g in CANONICAL_GLYPH_ORDER:
        per_patch = {}
        for r, c in glyph_coordinates(g):
            per_patch[(r // 3, c // 3)] = per_patch.get((r // 3, c // 3), 0) + 1
        assert sorted(per_patch) == glyph_patches(g)
        assert min(per_patch.values()) >= 2, (g, per_patch)
    assert len(glyph_patches("V")) == 6
    assert len(glyph_patches("A")) == 8
    assert len(glyph_patches("7")) == 8


def test_unknown_glyph_rejected():
    with pytest.raises(KeyError):
        glyph_coordinates("Z")


# ============================================================ milestone math
class _FakeCell:
    def __init__(self, cid, ff_src, weights, threshold=1000.0):
        self.id = cid
        self.ff_src = list(ff_src)
        self.acc_weights = list(weights)
        self.threshold = float(threshold)


def test_active_charge_sums_only_the_active_afferents():
    cell = _FakeCell("E", ["a", "b", "c"], [500.0, 400.0, 300.0])
    assert active_charge(cell, ["a", "b"]) == 900.0
    assert active_charge(cell, []) == 0.0
    assert active_charge(cell, ["a", "z"]) == 500.0


def test_one_event_maturity_boundary_is_exact():
    cell = _FakeCell("E", ["a", "b"], [500.0, 500.0])
    m = one_event_maturity(cell, ["a", "b"])
    assert m["active_charge"] == 1000.0 and m["margin"] == 0.0 and m["mature"] is True
    assert m["components"] == {"a": 500.0, "b": 500.0}
    # exactly one theta/2 afferent is NOT enough -- the detector must integrate two
    assert one_event_maturity(cell, ["a"])["mature"] is False


def test_throughput_tracker_uses_delivery_boundaries_as_denominator():
    tr = ThroughputTracker(window=50)
    # two source events coalescing onto ONE delivery boundary
    tr.note_source_event(10)
    tr.note_source_event(10)
    assert tr.note_boundary(11, True, tau=0.5) is True
    assert tr.note_boundary(12, True) is False          # not an eligible boundary
    rep = tr.report(min_boundaries=1)
    assert rep["eligible_source_events"] == 2
    assert rep["eligible_delivery_boundaries"] == 1
    assert rep["response_reliability"] == 1.0
    assert rep["median_tau"] == 0.5


def test_throughput_readiness_requires_both_count_and_reliability():
    tr = ThroughputTracker(window=50)
    for t in range(60):
        tr.note_source_event(t)
        tr.note_boundary(t + 1, True)
    assert tr.ready(min_boundaries=50, threshold=0.95) is True
    tr2 = ThroughputTracker(window=50)
    for t in range(60):
        tr2.note_source_event(t)
        tr2.note_boundary(t + 1, t % 2 == 0)
    assert tr2.ready(min_boundaries=50, threshold=0.95) is False
    tr3 = ThroughputTracker(window=50)
    for t in range(10):
        tr3.note_source_event(t)
        tr3.note_boundary(t + 1, True)
    assert tr3.ready(min_boundaries=50, threshold=0.95) is False   # too few boundaries


def test_c_readiness_needs_deposit_evidence_and_the_impulse_condition():
    ready = c_readiness(1000.0, 1000.0, deposits=3, spikes=2, updates=2)
    assert ready["ready"] is True and ready["impulse_one_shot"] is True
    # a mature weight with no committed deposit is NOT ready
    assert c_readiness(1000.0, 1000.0, deposits=0, spikes=0, updates=0)["ready"] is False
    # a deposit with a sub-threshold weight is NOT one-shot capable
    sub = c_readiness(999.0, 1000.0, deposits=5, spikes=1, updates=1)
    assert sub["impulse_one_shot"] is False and sub["ready"] is False
    # counters stay separate: no spike is not proof that no opportunity occurred
    assert sub["opportunities"] == 0 and sub["deposits"] == 5


# ============================================================ separability
def _sig(glyph, events, boundaries=10):
    return L3Signature(glyph=glyph, events=list(events),
                       taus=[0.5] * len(events), observed_boundaries=boundaries,
                       source_ids=["L", "R"])


def test_identical_traces_are_classified_unidentifiable():
    ev = [(2, "L", 500.0), (2, "R", 500.0), (5, "L", 500.0), (5, "R", 500.0),
          (8, "L", 500.0), (8, "R", 500.0)]
    out = classify_signatures({"V": _sig("V", ev), "A": _sig("A", ev), "7": _sig("7", ev)})
    assert out["verdict"] == "identical"
    assert out["identifiable"] is False
    assert out["source_collision"] is True
    assert all(p["exact_trace_match"] for p in out["pairs"])


def test_transient_only_difference_is_not_robust_separability():
    late = [(6, "L", 500.0), (6, "R", 500.0), (9, "L", 500.0), (9, "R", 500.0)]
    a = _sig("V", [(2, "L", 500.0)] + late)
    b = _sig("A", [(2, "R", 500.0)] + late)
    out = classify_signatures({"V": a, "A": b}, late_frac=0.4)
    assert out["verdict"] == "transient_only_separability"
    assert out["identifiable"] is False


def test_genuinely_different_late_summaries_are_separable():
    a = _sig("V", [(b, "L", 500.0) for b in (2, 4, 6, 8, 10)])
    b = _sig("A", [(b, "R", 500.0) for b in (2, 4, 6, 8, 10)])
    out = classify_signatures({"V": a, "A": b})
    assert out["verdict"] == "separable" and out["identifiable"] is True
    assert out["source_collision"] is False


def test_classification_needs_two_signatures():
    with pytest.raises(ValueError):
        classify_signatures({"V": _sig("V", [])})


def test_signature_summary_reports_the_loose_views():
    s = _sig("V", [(2, "L", 500.0), (2, "R", 500.0), (5, "L", 500.0)])
    summ = s.summary()
    assert summ["events_per_source"] == {"L": 2, "R": 1}
    assert summ["cooccurrence_by_boundary"] == {"L+R": 1, "L": 1}
    assert summ["inter_event_intervals"]["L"] == {"3": 1}


# ============================================================ taxonomy / slugs / resume
def test_failure_taxonomy_is_closed():
    rec = failure_record("representation_not_identifiable", first_seed=1)
    assert rec["taxon"] == "representation_not_identifiable" and rec["first_seed"] == 1
    with pytest.raises(ValueError):
        failure_record("made_up_reason")
    assert "timeout_unclassified" in FAILURE_TAXONOMY


def test_condition_slugs_are_deterministic_and_collision_checked():
    s = condition_slug("composition", condition="reference", B=5.0, seed=1)
    assert s == "composition-condition_reference-b_5p0-seed_1"
    reg = SlugRegistry()
    reg.claim(s)
    with pytest.raises(ValueError):
        reg.claim(s)


def test_resume_marker_requires_completed_status_and_matching_hash(tmp_path):
    d = str(tmp_path / "cell")
    os.makedirs(d)
    assert cell_is_complete(d, "abc") is False           # directory existence is not enough
    write_cell_status(d, status="running", config_hash="abc")
    assert cell_is_complete(d, "abc") is False
    write_cell_status(d, status="completed", config_hash="abc")
    assert cell_is_complete(d, "abc") is True
    assert cell_is_complete(d, "different") is False     # config drift is never resumed


def test_failed_cell_reruns_into_a_new_attempt_dir(tmp_path):
    base = str(tmp_path / "cell")
    assert next_attempt_dir(base) == base                # fresh cell uses the base dir
    os.makedirs(base)
    write_cell_status(base, status="failed", config_hash="abc")
    second = next_attempt_dir(base)
    assert second.endswith(".attempt2") and second != base


# ============================================================ engine contract
@pytest.fixture(scope="module")
def engine():
    return make_two_tower_engine(seed=1, B=REFERENCE_B)


def test_engine_is_built_under_the_declared_live_contract(engine):
    audit = assert_engine_contract(engine, B=REFERENCE_B)
    p = engine.params
    assert p["dual_fe_fes"] is True and p["dual_fe_B"] == 5.0
    assert p["eta"] == 4.0 and p["c_eta"] == 16.0
    assert p["leak_rate"] == 0.0 and p["refractory_steps"] == 0
    assert p["input_period"] == 0                        # AUTO pacing
    assert p["c_feedback_reset"] is True
    assert p["e_weight_cap_frac"] == 0.5                 # theta/2 detector ceiling
    assert p["eor_plasticity_enabled"] is False and p["eor_w_init_frac"] == 1.0
    assert audit["resolved_input_period"] == audit["feedback_loop_latency"] == 3
    assert audit["populations"] == {"ordinary": 168, "eor": 21, "c": 21}


def test_eor_is_a_fixed_non_plastic_relay_at_theta(engine):
    thr = float(engine.params["e_threshold"])
    n = 0
    for cell in engine.plastic:
        if engine._role_of.get(cell.id) != "Eor":
            continue
        n += 1
        assert cell.learn is False
        assert all(float(w) == thr for w in cell.acc_weights)
    assert n == 21


def test_detector_ceiling_is_theta_over_two_and_l3_needs_both_towers(engine):
    thr = float(engine.params["e_threshold"])
    index = FabricIndex(engine)
    l3_e = index.ordinary["L3"][index.l3]
    assert len(l3_e) == 8
    for nid in l3_e:
        cell = engine.exc[nid]
        # the ONLY thing L3 can see is two source identities
        assert sorted(cell.ff_src) == ["T0L2c00Eor", "T1L2c00Eor"]
        assert cell.w_cap == thr / 2.0
        # therefore a matured L3 detector needs BOTH towers in one delivery boundary
        assert one_event_maturity(cell, cell.ff_src)["threshold"] == thr


def test_fabric_index_selects_populations_by_metadata(engine):
    index = FabricIndex(engine)
    index.assert_builder_naming()
    assert len(index.l1_columns) == 18 and len(index.l2_columns) == 2
    assert index.l3 == "L3c00"
    assert set(index.towers) == {"T0L2c00", "T1L2c00"}
    assert all(len(v) == 9 for v in index.towers.values())
    assert index.dormant_c == ["L3c00"]                  # metadata, never an id special case
    for layer, expect in (("L1", 18), ("L2", 2), ("L3", 1)):
        assert len(index.ordinary[layer]) == expect
        assert all(len(v) == 8 for v in index.ordinary[layer].values())
    assert len(index.eor) == len(index.c) == len(index.i) == 21
    # tower membership is resolved from the parent chain, not the id prefix
    assert index.tower_of("T0L1c11") == "T0L2c00"
    assert index.tower_of("T1L1c20") == "T1L2c00"
    assert index.tower_of("T1L2c00") == "T1L2c00"


def test_active_selection_follows_the_glyph(engine):
    index = FabricIndex(engine)
    v = glyph_vector("V")
    active = index.active_l1_columns(v)
    assert len(active) == 6                              # three patches per tower
    assert len(index.active_child_eors("T0L2c00", v)) == 3
    assert len(index.active_child_eors("T1L2c00", v)) == 3
    seven = glyph_vector("7")
    assert len(index.active_l1_columns(seven)) == 8
    # a blank patch is genuinely blank: no active RGC at all
    blank = [c for c in index.l1_columns if c not in active]
    assert all(index.active_rgc_ids(c, v) == [] for c in blank)


def test_topology_fingerprint_is_stable_and_position_independent(engine):
    fp1, payload = topology_fingerprint(engine)
    other = make_two_tower_engine(seed=7, B=REFERENCE_B)   # different seed -> different pos
    fp2, _ = topology_fingerprint(other)
    assert fp1 == fp2                                      # display pos is excluded
    assert all("pos" not in n for n in payload["nodes"])
    assert json.dumps(payload, sort_keys=True)             # JSON-serializable


def test_engine_contract_assertion_catches_drift():
    from backend.simulation import SimulationEngine
    from backend.network_spec import two_tower_composition_spec as spec_fn
    bad = SimulationEngine(seed=1, topology="tiled_cc",
                           **{**ENGINE_CONTRACT, "eta": 0.01, "dual_fe_B": 5.0})
    bad.apply_topology(spec_fn())
    with pytest.raises(AssertionError):
        assert_engine_contract(bad, B=REFERENCE_B)


# ============================================================ built-in preset wiring
def test_two_tower_is_a_builtin_preset_everywhere():
    from backend.dashboard_config import CONFIG_SPEC
    from backend.network_spec import PRESETS
    from backend.simulation import VALID_TOPOLOGIES
    from backend import presets as ps
    assert "two_tower_composition" in PRESETS
    assert "two_tower_composition" in VALID_TOPOLOGIES
    assert "two_tower_composition" in ps.BUILTINS
    topo = next(c for c in CONFIG_SPEC if c["key"] == "topology")
    assert "two_tower_composition" in [o["value"] for o in topo["options"]]


def test_each_tiled_preset_declares_its_own_input_surface():
    from backend.network_spec import TILED_PRESETS, tiled_preset_input_size
    # the 9x9 family stays at 81; only the two-tower sheet is 162
    for name in ("tiled_cc", "tiled_cc_l1_4", "tiled_cc_direct_identity",
                 "tiled_cc_double_eor"):
        assert tiled_preset_input_size(name) == 81
    assert tiled_preset_input_size("two_tower_composition") == 162
    # every tiled preset must be in the table -- a new one cannot silently inherit 9x9
    for name in TILED_PRESETS:
        assert tiled_preset_input_size(name) > 0
    with pytest.raises(KeyError):
        tiled_preset_input_size("rg_coincidence")


def test_preset_engine_resolves_162_pixels_not_81():
    from backend.simulation import SimulationEngine
    e = SimulationEngine(seed=1, topology="two_tower_composition", cc_e_count=8)
    assert e.mode == "two_tower_composition"
    assert e.n_pix == 162
    assert (len(e.spec["nodes"]), len(e.spec["edges"])) == (393, 2162)
    t = e.topology()
    assert t["layers"] == ["RGC", "L1", "L2", "L3"]
    assert t["grid"] == {"rows": 9, "cols": 18}
    assert t["tiling"]["grid_shape"] == {"rows": 3, "cols": 6}
    # the topology-sized pattern bank never advertises a 9- or 81-length vector here
    assert all(len(v) == 162 for v in t["pattern_vectors"].values())


def test_preset_path_is_bit_identical_to_custom_topology_path():
    from backend.network_spec import two_tower_composition_spec
    from backend.simulation import SimulationEngine
    from experiments.basic_consolidation import plastic_edge_weights
    from experiments.two_tower_composition import ENGINE_CONTRACT
    for seed in (1, 3):
        viacustom = SimulationEngine(seed=seed, topology="tiled_cc", dual_fe_B=5.0,
                                     **ENGINE_CONTRACT)
        viacustom.apply_topology(two_tower_composition_spec(cc_e_count=8))
        viapreset = SimulationEngine(seed=seed, topology="two_tower_composition",
                                     dual_fe_B=5.0, **ENGINE_CONTRACT)
        assert viacustom.order == viapreset.order
        assert plastic_edge_weights(viacustom) == plastic_edge_weights(viapreset)
        assert all(viacustom.meta[n]["pos"] == viapreset.meta[n]["pos"]
                   for n in viacustom.order)


def test_n_pix_override_guard_uses_this_presets_own_surface():
    from backend.simulation import SimulationEngine
    # the 81-pixel assumption must NOT be applied to the 9x18 preset
    with pytest.raises(ValueError, match="9x18"):
        SimulationEngine(seed=1, topology="two_tower_composition", n_pix=81)
    ok = SimulationEngine(seed=1, topology="two_tower_composition", n_pix=162)
    assert ok.n_pix == 162
    # and the 9x9 family still rejects 162
    with pytest.raises(ValueError, match="9x9"):
        SimulationEngine(seed=1, topology="tiled_cc", n_pix=162)


def test_preset_store_resolves_the_two_tower_dims(tmp_path, monkeypatch):
    from backend import presets as ps
    monkeypatch.setattr(ps, "PRESET_DIR", str(tmp_path))
    listed = {p["name"]: p for p in ps.list_presets(9, 8)}
    assert listed["two_tower_composition"]["builtin"] is True
    # loading it must use ITS 162-pixel surface, never the active engine's n_pix
    spec = ps.load_spec("two_tower_composition", 9, 8)
    assert len(spec["nodes"]) == 393
    assert spec["topology"]["input_shape"] == {"rows": 9, "cols": 18}
    assert all(n.get("pos") is not None for n in spec["nodes"])   # editor-visible layout


def test_glyphs_are_selectable_stimuli_on_the_live_preset():
    from backend.simulation import SimulationEngine
    e = SimulationEngine(seed=1, topology="two_tower_composition", cc_e_count=8)
    t = e.topology()
    # the four 3x3 locals keep their order; the glyphs are appended
    assert t["patterns"] == ["row 1", "col 1", "diag \\", "diag /", "V", "A", "7"]
    assert t["tiling"]["sheet_patterns"] == ["V", "A", "7"]
    assert all(len(v) == 162 for v in t["pattern_vectors"].values())
    # each glyph drives its own full-sheet vector, matching the experiment's definition
    for g in CANONICAL_GLYPH_ORDER:
        e.set_pattern(g)
        assert list(map(int, e.input_vec)) == glyph_vector(g)
        assert e.current_pattern == g
        # a whole-sheet stimulus is not attributable to any one patch
        assert e.patch_pattern_map() == []


def test_glyph_bank_is_keyed_on_input_shape_not_preset_name():
    from backend.network_spec import sheet_glyph_bank, sheet_glyph_names
    from backend.simulation import SimulationEngine
    assert sheet_glyph_names(9, 18) == ("V", "A", "7")
    assert sheet_glyph_names(9, 9) == ()            # the 9x9 family has no whole-sheet bank
    assert sheet_glyph_bank(9, 9) == {}
    for name in ("tiled_cc", "tiled_cc_l1_4", "rg_coincidence"):
        t = SimulationEngine(seed=1, topology=name).topology()
        if name == "rg_coincidence":
            assert "tiling" not in t              # non-tiled graphs carry no tiling block
        else:
            assert t["tiling"]["sheet_patterns"] == []
        assert "V" not in t["patterns"]
    # fresh objects every call (no shared mutable stimulus vectors)
    a, b = sheet_glyph_bank(9, 18), sheet_glyph_bank(9, 18)
    assert a == b and a["V"] is not b["V"]
    a["V"][0] = 0
    assert b["V"] != a["V"]


def test_glyph_cannot_be_embedded_into_a_single_patch():
    from backend.simulation import SimulationEngine
    e = SimulationEngine(seed=1, topology="two_tower_composition")
    with pytest.raises(KeyError, match="WHOLE-SHEET"):
        e.set_patch_pattern(0, 0, "V")
    # and choosing a glyph after a patch composition replaces it rather than merging
    e.set_patch_pattern(0, 0, "row 1")
    e.set_patch_pattern(2, 5, "col 1")
    assert len(e.patch_pattern_map()) == 2
    e.set_pattern("7")
    assert e.patch_pattern_map() == []
    assert int(e.input_vec.sum()) == 26


def test_live_patch_patterns_work_across_all_eighteen_patches():
    from backend.simulation import SimulationEngine
    e = SimulationEngine(seed=1, topology="two_tower_composition", cc_e_count=8)
    e.set_patch_pattern(0, 0, "row 1")
    e.set_patch_pattern(2, 5, "diag \\")        # far corner of the RIGHT tower
    assert len(e.patch_pattern_map()) == 2
    assert int(e.input_vec.sum()) == 6          # three pixels per 3x3 patch, disjoint
    for _ in range(30):
        e.step()


def test_short_run_produces_column_winners_at_every_layer(engine):
    from experiments.two_tower_composition import make_two_tower_engine as mk
    eng = mk(seed=1, B=REFERENCE_B)
    index = FabricIndex(eng)
    eng.set_input(glyph_vector("V"))
    layers_seen = set()
    for _ in range(400):
        dyn = eng.step()
        for cid in dyn.get("column_winners", {}):
            layers_seen.add(index.columns[cid]["layer"])
    assert layers_seen == {"L1", "L2", "L3"}
    # inactive L1 columns stay silent under a glyph that does not drive them
    v = glyph_vector("V")
    silent = [c for c in index.l1_columns if not index.active_rgc_ids(c, v)]
    assert silent and all(eng._spike_hist[index.ordinary["L1"][c][0]].count(1) == 0
                          for c in silent)
