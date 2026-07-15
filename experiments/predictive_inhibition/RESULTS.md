# Local Predictive Inhibition — Results

Run `pi_20260714-201812`, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

## Preregistered primary outcome (Section 13)

**Supportive: True**  (M_live=0.0518, M_shuffled=-0.0063)

| Criterion | Result | Pass |
| --- | --- | --- |
| 1. median CSC(live) > 0 | 0.0518 | True |
| 2. live > local_only | 10/10 (need >=8) | True |
| 3. live > shuffled | 10/10 (need >=8) | True |
| 4. M_shuffled ≤ 0.75·M_live | — | True |
| 5. reversal recovery | 10/10 (need >=7) | True |
| 6. no permanent L1E silence | silent seeds=[] | True |

## Contextual task — CSC_primary (last 100 acquisition)

| Mode | mean | std | median | min | max | recovered |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/10 |
| legacy_feedback | 0.0411 | 0.0225 | 0.0388 | 0.0033 | 0.0908 | 10/10 |
| local_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/10 |
| local_plus_feedback | 0.0464 | 0.0173 | 0.0518 | 0.0049 | 0.0668 | 10/10 |
| time_shuffled_feedback | -0.0055 | 0.0086 | -0.0063 | -0.0181 | 0.0109 | 10/10 |

### Per-seed CSC_primary

| seed | baseline | legacy_feedback | local_only | local_plus_feedback | time_shuffled_feedback |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 0.0507 | 0.0000 | 0.0556 | -0.0007 |
| 1 | 0.0000 | 0.0553 | 0.0000 | 0.0549 | -0.0092 |
| 2 | 0.0000 | 0.0368 | 0.0000 | 0.0303 | -0.0181 |
| 3 | 0.0000 | 0.0526 | 0.0000 | 0.0500 | -0.0063 |
| 4 | 0.0000 | 0.0033 | 0.0000 | 0.0454 | 0.0109 |
| 5 | 0.0000 | 0.0408 | 0.0000 | 0.0536 | -0.0062 |
| 6 | 0.0000 | 0.0263 | 0.0000 | 0.0049 | -0.0089 |
| 7 | 0.0000 | 0.0191 | 0.0000 | 0.0641 | 0.0049 |
| 8 | 0.0000 | 0.0908 | 0.0000 | 0.0668 | -0.0181 |
| 9 | 0.0000 | 0.0355 | 0.0000 | 0.0385 | -0.0036 |

## Inhibitory charge & firing (contextual)

| Mode | total charge (median) | L1I spikes (median) |
| --- | --- | --- |
| baseline | 0 | 0 |
| legacy_feedback | 11245500 | 33736 |
| local_only | 0 | 0 |
| local_plus_feedback | 5966258 | 12337 |
| time_shuffled_feedback | 4626347 | 10748 |

## Predictor contrasts (PWD) & L1I event-class spike probability

| Mode | PWD_F (mean) | PWD_G (mean) | P(spike\|coincident) | P(spike\|feedback_only) |
| --- | --- | --- | --- | --- |
| baseline | 0.0 | 0.0 | None | None |
| legacy_feedback | 0.0 | 0.0 | 0.9365 | 0.9327 |
| local_only | 0.0 | 0.0 | None | None |
| local_plus_feedback | -1337.9 | -1342.5 | 0.9681 | 0.0051 |
| time_shuffled_feedback | -280.3 | 370.0 | 0.9242 | 0.0485 |

## Four-pattern task — distinct L2 owners & charge

| Mode | distinct owners (mean) | total charge (median) |
| --- | --- | --- |
| baseline | 3.20 | 0 |
| legacy_feedback | 3.20 | 4594500 |
| local_only | 3.20 | 0 |
| local_plus_feedback | 3.20 | 2164000 |
| time_shuffled_feedback | 3.40 | 1841364 |

## Shuffle checks (Section 9)

- seeds checked: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- methods used: ['derangement_try1', 'derangement_try2', 'derangement_try3', 'derangement_try4', 'derangement_try6']
- per-source counts AND all-zero-vector count preserved everywhere: True

## Direct answers (Section 15.7)

- **predictors feature specific:** yes — predictor rows differentiated by source; PWD_F mean=-1337.9343, PWD_G mean=-1342.5013 (sign reflects the suppression->trace feedback: succeeding suppression lowers the paired L1E trace and drives the predictor down on the suppressed feature)
- **selectivity contextual not frequency:** yes — live CSC median 0.0518 > 0 and exceeds the rate-matched shuffled control (-0.0063); the CSC contrast already fixes feature frequency (same feature active in both terms)
- **shuffling removed it:** yes — M_shuffled/M_live = -0.1206 (criterion needs <= 0.75); shuffled median -0.0063
- **suppression and predictor reversed:** partial — reversal CSC median -0.0339 vs acquisition 0.0518; live recovery 10/10 (need >=7)
- **l2 specialization independent of firing:** yes — distinct L2 owners are similar across modes despite very different L1I firing (four-pattern distinct owners: live=3.2, legacy=3.2, baseline=3.2); the L1 feedback loop does not drive L2 competition
- **What failed:** no criterion failed

## Parameter registry (resolved, local_plus_feedback)

```json
{
  "paired_local_enabled": true,
  "predictive_feedback_enabled": true,
  "l2_to_l1i_delivery_enabled": true,
  "predictive_local_weight_frac": 0.4,
  "predictive_feedback_init_frac": 0.1,
  "predictive_feedback_eta_up": 0.08,
  "predictive_feedback_eta_down": 0.04,
  "predictive_trace_tau_steps": 2.0,
  "predictive_l1i_leak_rate": 0.5,
  "predictive_output_gate_frac": 0.5,
  "threshold": 1000,
  "threshold_l2": 8000,
  "l2i_threshold_frac": 0.3333333333333333,
  "l1i_threshold_frac": 1,
  "refractory": 1,
  "l2_charge_chunks": 20,
  "competitive_weight_update": "redistribution",
  "theta_L1I_resolved": 2666.6666666666665,
  "G_resolved": 2666.6666666666665,
  "local_weight_resolved": 1066.6666666666667,
  "feedback_init_resolved": 266.6666666666667,
  "L1E_gate_resolved": -500.0,
  "trace_lambda": 0.6065306597126334
}
```