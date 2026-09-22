"""Asset bookkeeping for the five benchmark tasks (Sec. 5.1, App. A).

The paper builds on two publicly released environment suites:

* the Allegro-Kuka manipulation tasks (an Allegro hand with 16 DoF mounted on a
  7-DoF Kuka arm) of Petrenko et al. (2023), and
* the 24-DoF Shadow Hand / 16-DoF Allegro Hand in-hand reorientation tasks of
  Li et al. (2023) (originally from OpenAI et al. 2018).

The asset files themselves (URDFs, meshes) are *not* vendored into this
repository.  Point ``cfg.env.asset_root`` (or the ``SAPG_ASSET_ROOT``
environment variable) at a directory laid out as below.
"""

from __future__ import annotations

import os
from typing import Dict

DEFAULT_ASSET_ROOT = os.environ.get(
    "SAPG_ASSET_ROOT", os.path.join(os.path.dirname(__file__), "assets")
)

# Relative paths inside the asset root.
ALLEGRO_KUKA_ASSETS: Dict[str, str] = {
    "kuka": "kuka_allegro/kuka_description.urdf",
    "allegro": "kuka_allegro/allegro_hand_description_right.urdf",
    "combined": "kuka_allegro/kuka_allegro.urdf",
    "object": "objects/box.urdf",
    "table": "objects/table.urdf",
    "bucket": "objects/bucket.urdf",
}

IN_HAND_ASSETS: Dict[str, str] = {
    "shadow_hand": "shadow_hand/shadow_hand.urdf",
    "allegro_hand": "allegro_hand/allegro_hand.urdf",
    "object": "objects/block.urdf",
}


def asset_path(asset_root: str, key: str, suite: str = "allegro_kuka") -> str:
    table = ALLEGRO_KUKA_ASSETS if suite == "allegro_kuka" else IN_HAND_ASSETS
    if key not in table:
        raise KeyError(f"Unknown asset '{key}' for suite '{suite}'")
    return os.path.join(asset_root, table[key])
