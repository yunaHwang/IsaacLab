"""Extract per-episode scene info for the 300 stack_2_cubes episodes (600-899) of
Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi into dp-fm-ood, so run_policy_fm_0910_pi.py can
recreate any episode's scene in Isaac Lab without the dataset download.

    conda activate lerobot_0.6.1_multitask_dit     (any env with pandas + pyarrow)
    python extract_scrape_stack2cubes_episodes.py

Writes data/scrape_so101_stack2cubes/:
    raw/                         the dataset files this reads (downloaded if missing)
    episodes.json                {episode: instruction, top/base color names + rgb, light scale,
                                  top/base cube xy (robot frame, m), length}
    episodes.npz                 per episode: state_<ep>, action_<ep> ([T, 6] LeRobot normalized
                                 motor units, the units Pi0.5 consumes/produces)

Cube positions come from SCRAPE's own perception labels, not from the gripper path:
    top  cube xy = subtask.target_position while the subtask is on the top block
    base cube xy = skill.goal_position of the "Stack <top> block on <base> block" skill
Cube yaw is not recorded anywhere in the dataset.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://huggingface.co/datasets/Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi/resolve/main"
HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=str(HERE / "data" / "scrape_so101_stack2cubes"))
args = parser.parse_args()
OUT = Path(args.out_dir)
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch(rel):
    local = RAW / rel.replace("/", "__")
    if not local.exists():
        print(f"downloading {rel}")
        subprocess.run(["curl", "-sfL", "-o", str(local), f"{REPO}/{rel}"], check=True)
    return local


meta = {}
for line in open(fetch("meta/scrape_episode_metadata.jsonl")):
    m = json.loads(line)
    if m["multitask_task_id"] == "stack_2_cubes":
        meta[m["dataset_episode_index"]] = m
episodes = pd.read_parquet(fetch("meta/episodes/chunk-000/file-000.parquet")).set_index("episode_index")
fetch("meta/info.json")
fetch("meta/stats.json")

file_ids = sorted({int(episodes.loc[ep, "data/file_index"]) for ep in meta})
cols = ["episode_index", "frame_index", "observation.state", "action", "skill.natural_language",
        "skill.goal_position.robot_xyzrpy", "subtask.object_name", "subtask.target_position"]
df = pd.concat([pd.read_parquet(fetch(f"data/chunk-000/file-{fi:03d}.parquet"), columns=cols) for fi in file_ids])

info, arrays = {}, {}
for ep, m in sorted(meta.items()):
    e = df[df.episode_index == ep].sort_values("frame_index")
    app = m["appearance"]
    top_name, base_name = app["stack_expected_calls"][0]            # e.g. ("gray block", "pink block")
    top_color, base_color = top_name.split()[0], base_name.split()[0]
    objs = {o["color_name"]: o for o in app["objects"].values()}

    tgt = np.stack(e["subtask.target_position"])
    on_top = (e["subtask.object_name"].values == top_name) & (np.abs(tgt).sum(1) > 0)
    top_xy = tgt[on_top][0, :2] if on_top.any() else None

    stack_re = re.compile(rf"^stack {top_color} block on {base_color} block$", re.I)
    is_stack = np.array([bool(stack_re.match(s.strip())) for s in e["skill.natural_language"].values])
    base_xy = np.stack(e["skill.goal_position.robot_xyzrpy"])[is_stack][0, :2] if is_stack.any() else None

    info[str(ep)] = {
        "instruction": m["instruction"],
        "top_color": top_color, "top_rgb": objs[top_color]["rgb"],
        "base_color": base_color, "base_rgb": objs[base_color]["rgb"],
        "light_scale": app["light_scale"],
        "top_cube_xy": None if top_xy is None else [round(float(v), 4) for v in top_xy],
        "base_cube_xy": None if base_xy is None else [round(float(v), 4) for v in base_xy],
        "length": int(len(e)),
    }
    arrays[f"state_{ep}"] = np.stack(e["observation.state"]).astype(np.float32)
    arrays[f"action_{ep}"] = np.stack(e["action"]).astype(np.float32)

(OUT / "episodes.json").write_text(json.dumps(info, indent=1))
np.savez_compressed(OUT / "episodes.npz", **arrays)
missing = [ep for ep, v in info.items() if v["top_cube_xy"] is None or v["base_cube_xy"] is None]
print(f"wrote {len(info)} episodes to {OUT} (cube positions missing for {len(missing)}: {missing[:10]})")
print("example:", json.dumps(info[next(iter(info))]))
