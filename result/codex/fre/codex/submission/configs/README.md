# Configurations

Each JSON file is a set of `TrainConfig` overrides accepted by
`train_fre.py --config <file>`.  CLI flags take precedence over the file, and
explicit values in the file take precedence over the domain defaults.

| File | Paper section |
| --- | --- |
| `antmaze_fre_all.json` | §5.2 / §5.3 — the vanilla equal mixture of goal / linear / MLP rewards |
| `antmaze_fre_goals.json` | §5.3 — goal-reaching rewards only |
| `antmaze_fre_lin.json` | §5.3 — random linear rewards only |
| `antmaze_fre_mlp.json` | §5.3 — random MLP rewards only |
| `antmaze_fre_lin_mlp.json` | §5.3 — linear + MLP |
| `antmaze_fre_goal_mlp.json` | §5.3 — goal-reaching + MLP |
| `antmaze_fre_goal_lin.json` | §5.3 — goal-reaching + linear |
| `antmaze_fre_hint.json` | §5.4 — prior augmented with unit-direction rewards |
| `exorl_walker_fre_all.json`, `exorl_cheetah_fre_all.json` | §5.2 ExORL columns |
| `exorl_walker_fre_hint.json`, `exorl_cheetah_fre_hint.json` | §5.4 — prior augmented with specific-velocity rewards |
| `kitchen_fre_all.json` | §5.2 Kitchen column |

Example:

```bash
python train_fre.py --config configs/antmaze_fre_goal_lin.json --seed 0
```

The AntMaze configs pin the Appendix A schedule (150k encoder + 850k policy
steps); the ExORL and Kitchen configs leave the schedule unset so that the
domain defaults (1M + 1M steps) apply.
