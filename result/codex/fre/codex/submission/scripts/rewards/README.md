# Adapter for `controllable_agent`

`controllable_agent` computes rewards through a task object with a
`get_reward(physics)` method, sampled 5120 times at evaluation time to fit the
linear (FB/SF) reward decoder.  The paper replaced the default DMC / D4RL
rewards with its custom evaluation tasks; the same tasks are implemented here in
`fre/tasks/`.

To wire them together, subclass the environment wrapper in
`controllable_agent` and replace `get_reward` with a call into the
corresponding task, e.g.:

```python
import numpy as np
from fre.tasks.antmaze import make_antmaze_goal_tasks
from fre.tasks.exorl import make_velocity_tasks


class CustomRewardTask:
    """controllable_agent-compatible task exposing the paper's rewards."""

    def __init__(self, task, obs_fn):
        self._task = task
        self._obs_fn = obs_fn  # physics -> observation vector

    def get_reward(self, physics):
        obs = self._obs_fn(physics).reshape(1, -1)
        return float(np.asarray(self._task.reward(obs)).reshape(-1)[0])
```

Then build the task list from the same suites used for FRE:

```python
from fre.tasks.antmaze import make_antmaze_suites

suites = make_antmaze_suites()
eval_tasks = {s.name: s.tasks for s in suites}
```

The evaluation protocol in `controllable_agent` (5120 reward samples, mean over
20 episodes, 5 seeds) already matches the paper, so no further changes are
required.
