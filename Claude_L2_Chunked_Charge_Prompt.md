# Claude L2 Chunked Charge Prompt

Please inspect the current L2 timestep / winner-processing logic, especially
`backend/simulation.py`, `neuron_flexible.py`, and the methodology docs.

Goal: implement a consolidation-first L2 competition refinement by increasing
the temporal granularity of L2 feedforward charge arrival inside a frozen outer
timestep.

Conceptual flow:

1. During an outer timestep, L1 produces its usual spike vector.
2. Do not advance the global simulation clock for substeps.
3. For each L2E neuron, compute the feedforward drive it would normally receive
   from the current L1 spike vector:

   ```text
   full_drive_j = sum_i x_i * w_ji
   ```

4. Split that drive into K chunks:

   ```text
   chunk_drive_j = full_drive_j / K
   ```

5. Inside the same outer timestep, run an inner L2-only chunk loop:
   - add `chunk_drive_j` to each L2E
   - check all L2E thresholds
   - if one or more L2E neurons crossed threshold, choose the max-charge neuron
     as the single winner
   - fire that winner
   - send the winner spike to L2I
   - if L2I fires, inhibit every other L2E neuron through the learned
     L2I->L2E inhibitory gate
   - stop processing remaining chunks for this L2 competition event
6. Then continue the rest of the normal timestep: feedback, L1I,
   update/leak/refractory bookkeeping, logging, etc.

Important constraints:

- This is not K full timesteps. It is a frozen-timestep inner integration loop
  for L2 feedforward charge only.
- Do not advance `self.timestep` inside chunks.
- Do not apply normal full-neuron `update()` or leak between chunks unless the
  current architecture makes that unavoidable. Prefer no substep leak for the
  first ablation.
- The chunk amount is based on each specific current synaptic weight divided by
  K.
- Preserve simultaneous L1 volley semantics: all active inputs contribute their
  weight/K during each chunk. Do not invent an ordering among pixels.
- Keep the existing max-charge argmax winner selection among threshold-crossers.
- Keep lateral inhibition pool-wide: inhibit every non-winning L2E, not only
  the other threshold-crossers.
- Add this behind a config parameter such as `l2_charge_chunks`, default `1`,
  where `1` exactly preserves current behavior.
- Do not CUDA/GPU-optimize this. Keep it CPU/Numpy/Python for now. The network
  is small enough for this algorithmic ablation, and CUDA would add debugging,
  determinism, and dashboard serialization complexity before it is needed.
- Preserve the existing one-render-frame-per-outer-timestep dashboard behavior.
  Do not emit per-chunk websocket frames. At most expose optional diagnostics
  such as `l2_charge_chunks` and `l2_winner_chunk`.
- Do not implement distance weighting.
- Do not add membrane noise.
- Keep weight initialization uniform/random.

Please first verify the current procedure, then make the smallest scoped
implementation needed for this ablation, update the relevant docs, and add or
adjust tests or a small harness check showing `l2_charge_chunks=1` preserves
baseline behavior while larger values run the chunked inner loop.
