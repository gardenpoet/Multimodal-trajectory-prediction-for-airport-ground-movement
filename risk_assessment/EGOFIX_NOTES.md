# Ego-identity fix for the continuous (dense per-frame) risk series

## Problem

Each frame of the b3/s70 and b7/s62 dense 50-frame continuous-prediction
series is an independent inference call. `ego_id` always defaults to the
dataset's own auto-detected slot 0, and `ref_agent_idx` auto-detects
"whichever agent achieves the closest approach" within that frame's own
future window — both recomputed independently at every frame, with no
persistent agent identity across frames. As a result, the physical
aircraft behind the "ego" label can drift across the window.

Position-continuity matching (exhaustive nearest-neighbour matching
against every agent slot in each frame, seeded from each case's
validated headline frame — frame 208 for b3/s70, frame 2194 for b7/s62)
confirmed:

- **b7/s62**: frames 2176–2196 are already correct (ego = the headline
  aircraft). From frame 2197 onward, the dataset's own auto-detected
  ego/interactive pairing cleanly swaps to the *other* of the same two
  real aircraft (frame 2196's ego matches frame 2197's auto-detected
  interactive agent to within 0.7 m).
- **b3/s70**: frames 206–221 are already correct. Frames 184–205 track a
  **different pair of aircraft entirely** — this case's own two
  protagonists are present only as background ("other_N") agents there.
  Frames 222–233 are a clean role swap, same as b7/s62's.

## Fix

Rather than relabel (which only swaps which of the two real aircraft is
treated as "ego," still breaking continuity for the one we actually
want), these `_egofix` scripts **pin** `+data.dataset.config.ego_agent_id=<agent_idx>`
to the scene-local index of the correct physical aircraft at each
affected frame, read directly from that frame's own (already-downloaded)
`agent_idx` field. `case_ref_agent_idx` is deliberately **left unset** in
every `_egofix` script, so the interactive/reference agent continues to
auto-detect to whichever other aircraft is closest — only ego's own
identity is required to stay fixed; the interactive side is allowed to
vary frame to frame, same as the already-correct segments.

`agent_idx` is **not** a persistent track ID across different per-frame
queries of the same scene recording (the dataset re-derives agent
ordering fresh each time, and the scene's own agent count changes as
aircraft enter/exit — see the `_n-N` suffix in `scene_file`), so this
pinned value is specific to each individual frame's own query and was
computed once by the matching script, not guessed.

## Frames fixed

- `b3s70`: 184–205 (agent_idx 3, 1, or 2 depending on sub-range) and
  222–233 (agent_idx 1). Frames 206–221 need no fix.
- `b7s62`: 2197–2224 (agent_idx 1). Frames 2174, 2176–2196 need no fix.

## Running

```
sbatch risk_assessment/run_case_risk_dynamics_kbos_<case>_f<frame>_<1T|2T|4T>_egofix.sh
sbatch risk_assessment/run_case_trajectories_kbos_<case>_f<frame>_stgcnn_egofix.sh
sbatch risk_assessment/run_case_trajectories_kbos_<case>_f<frame>_amelia_baseline_egofix.sh
```

or submit everything at once:

```
bash risk_assessment/submit_egofix_all.sh
```

Each writes to `.../risk_assessment/out/kbos_case_<case>_f<frame>_<kind>_egofix.json`
(note the `_egofix` suffix — these do **not** overwrite the original,
drifting-identity outputs already downloaded). Once downloaded locally
to the same `HPC runs/timeline` folder as the originals, the
`_egofix` files should replace frames 184–205 and 222–233 (b3/s70) and
2197–2224 (b7/s62) when rebuilding `grid_data_paperstyle_*.json` and
`continuous_timeline_data_*.json`, giving both cases a single
consistent ego identity across the full 50-frame window while the
interactive agent is left free to vary, matching b762/s32 (which
already needed no fix).
