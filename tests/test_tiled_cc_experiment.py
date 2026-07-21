"""Regression coverage for the headless tiled_cc acceptance experiment: the eleven
mechanical isolation checks pass, there are zero cross-column reset violations and zero
top-C deposits, and the two-patch probe shows two independent L1 winners under a single
hard L2 winner. This keeps the authoritative acceptance evidence reproducible.
"""

from experiments.tiled_cc_experiment import run_isolation, run_two_patch


def test_center_patch_isolation_all_checks_pass():
    iso = run_isolation(seed=1, cc_e_count=8, patch=(1, 1), pattern='row 1',
                        boundaries=280, leak_rate=0.0)
    assert iso['cross_column_reset_violations'] == 0
    assert iso['top_c_deposits'] == 0 and iso['top_c_spikes'] == 0
    # the full mechanical chain matured within the run
    for key in ('4_l1_e_reaches_eor', '6_eor_reaches_l2', '7_l2_hard_wta',
                '8_l2_apical_to_all_nine_c_but_only_eligible_deposits',
                '10_top_l2_c_dormant', '11_no_inactive_column_reset'):
        assert iso['acceptance_checks'][key], key
    assert iso['all_checks_pass']


def test_isolation_active_column_is_the_only_learner():
    iso = run_isolation(boundaries=280)
    active = iso['resolved']['active_column']
    # only the active column's C ever deposits
    for cid, t in iso['per_column'].items():
        if cid != active:
            assert t['c_deposits'] == 0, cid


def test_two_patch_independent_columns_under_single_l2_winner():
    two = run_two_patch(seed=1, cc_e_count=8, patches=((0, 0), (2, 2)),
                        pattern='row 1', boundaries=80, leak_rate=0.0)
    assert two['two_independent_l1_winners'] is True
    assert two['l2_stays_single_winner'] is True
    assert two['max_l2_winners_in_a_boundary'] <= 1
