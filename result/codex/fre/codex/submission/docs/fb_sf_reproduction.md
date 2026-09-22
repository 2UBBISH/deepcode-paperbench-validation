# Reproducing the FB and SF columns of Table 1

The addendum specifies that the Forward-Backward (FB) and Successor-Features
(SF) baselines must be trained and evaluated with
[`facebookresearch/controllable_agent`](https://github.com/facebookresearch/controllable_agent).
Re-implementing them here would be both redundant and contrary to the
instructions, so this repository instead:

1. provides `scripts/run_fb_sf.sh`, which clones and drives that codebase;
2. documents every modification the paper's authors made to it (below);
3. ships the paper's evaluation reward functions in `scripts/rewards/` in a
   form that can be dropped into `controllable_agent`'s environment wrappers.

## What the original authors did

From the addendum:

* All SF/FB ExORL experiments use the **RND** dataset.
* **ICM** features are used for SF.
* Training the FB/SF policies did **not** require any changes to the
  `controllable_agent` codebase.
* For evaluation, the set of evaluation tasks in the paper was re-implemented
  by *introducing a custom reward function into the pre-existing environments*
  (antmaze, walker, cheetah, kitchen) that replaced the default reward with the
  paper's custom rewards.
* FB/SF are given **5120** reward samples at evaluation time (rather than the 32
  used by FRE), "to be consistent with prior work".
* Evaluation numbers are logged during the training run.

## Step-by-step

```bash
# 0) set up the upstream codebase
git clone https://github.com/facebookresearch/controllable_agent
cd controllable_agent
# follow its README to install dependencies

# 1) download the offline RND datasets (ExORL) and the D4RL datasets
#    (AntMaze / Kitchen, pre-June-2024 revision)

# 2) build the replay buffer exactly as described in the upstream README

# 3) launch training and record the evaluation numbers logged during training
python train_FB.py --env_name=antmaze-large-diverse-v2 --eval_tasks=antmaze
python train_SF.py --env_name=antmaze-large-diverse-v2 --eval_tasks=antmaze --use_icm=1
python train_FB.py --env_name=walker-rnd --eval_tasks=exorl
python train_SF.py --env_name=walker-rnd --eval_tasks=exorl --use_icm=1
```

## Custom evaluation rewards

The reward functions used at evaluation are exactly the ones implemented in
`fre/tasks/`, so rather than duplicating them inside the upstream repository we
point the upstream environment wrappers at the same definitions:

| Column of Table 1 | Reward implementation |
| --- | --- |
| `ant-goal-reaching` | `fre.tasks.antmaze.make_antmaze_goal_tasks` |
| `ant-directional` | `fre.tasks.antmaze.make_antmaze_directional_tasks` |
| `ant-random-simplex` | `fre.tasks.antmaze.make_antmaze_simplex_tasks` |
| `ant-path-*` | `fre.tasks.antmaze.make_antmaze_path_tasks` |
| `exorl-*-velocity` | `fre.tasks.exorl.make_velocity_tasks` |
| `exorl-*-goals` | `fre.tasks.exorl.make_goal_tasks` |
| `kitchen` | `fre.tasks.kitchen.make_kitchen_suite` |

`scripts/rewards/README.md` explains the small adapter that
`controllable_agent` needs (a `get_reward(physics)` override that delegates to
the corresponding task object).
