"""Save what the scene looked like during data generation - cameras, lights, materials/textures, randomization
events - plus the full env config, as sidecar files next to the generated dataset.

Import AFTER the AppLauncher block (it reads the USD stage via pxr/omni). Used by generate_dataset_seeded.py:

    write_run_metadata(env, output_file, seed=..., task=...)   # once, after the first env.reset()
    log_episodes(env, output_file)                              # one line per episode written to the hdf5(s)

For --output_file ./datasets/visuomotor-based/0901_topup_seed4_30.hdf5 this writes
    ./datasets/visuomotor-based/0901_topup_seed4_30_env.json        cameras, lights, materials, events, seed, sim
    ./datasets/visuomotor-based/0901_topup_seed4_30_env_cfg.yaml    full env config (Isaac Lab's dump_yaml)
    ./datasets/visuomotor-based/0901_topup_seed4_30_episodes.jsonl  one line per dataset episode: file + demo_N,
                                                                     success, cameras, lights, materials at its reset

Values come from the stage, not just the config, so they reflect what was rendered: e.g. this task's
randomize_scene_lighting_domelight / randomize_visual_texture_material events only randomize when
env.cfg.eval_mode is on and otherwise write the defaults - both the event definitions and the resulting light and
material values are recorded.
"""

import json
import os
import time
from pathlib import Path

def _jsonable(value):
    """USD / torch / numpy values -> plain JSON types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "path") and value.__class__.__name__ == "AssetPath":  # Sdf.AssetPath
        # resolvedPath turns layer-relative texture paths ("Textures/x.png") into the actual file/URL
        return getattr(value, "resolvedPath", "") or value.path
    if hasattr(value, "tolist"):  # torch / numpy
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:  # Gf vectors / matrices / quaternions
        return [float(v) for v in value]
    except TypeError:
        return str(value)


def _on_articulation(env, sensor):
    """True for cameras mounted under a robot/articulation prim (e.g. wrist_cam under Robot/panda_hand). Their world
    pose moves every step, and reading it right after env.reset() is stale: Camera.reset updates the pose before the
    reset events move the robot to its new start joints. So for those only the fixed mount offset is logged."""
    return any(sensor.cfg.prim_path.startswith(art.cfg.prim_path + "/") for art in env.scene.articulations.values())


def _cameras(env):
    out = {}
    origin = env.scene.env_origins[0]
    for name, sensor in env.scene.sensors.items():
        data = getattr(sensor, "data", None)
        if data is None or getattr(data, "intrinsic_matrices", None) is None:
            continue
        cfg = sensor.cfg
        spawn = cfg.spawn
        out[name] = {
            "prim_path": cfg.prim_path,
            "width": cfg.width,
            "height": cfg.height,
            "data_types": list(cfg.data_types),
            "update_period": cfg.update_period,
            "spawn": {k: _jsonable(getattr(spawn, k)) for k in (
                "focal_length", "focus_distance", "horizontal_aperture", "vertical_aperture",
                "horizontal_aperture_offset", "vertical_aperture_offset", "clipping_range", "projection_type",
                "f_stop", "lock_camera") if hasattr(spawn, k)},
            "offset": {"pos": _jsonable(cfg.offset.pos), "rot_wxyz": _jsonable(cfg.offset.rot),
                       "convention": cfg.offset.convention},
            "intrinsic_matrix": _jsonable(data.intrinsic_matrices[0]),
        }
        if _on_articulation(env, sensor):
            out[name]["world_pose"] = "not logged: moves with the robot (see offset; per-step pose follows obs/eef_pos, eef_quat)"
        else:
            out[name].update({"pos_in_env": _jsonable(data.pos_w[0] - origin),
                              "quat_w_ros_wxyz": _jsonable(data.quat_w_ros[0]),
                              "quat_w_world_wxyz": _jsonable(data.quat_w_world[0])})
    return out


def _lights(stage):
    from pxr import UsdLux

    lights = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        attrs = {a.GetName(): _jsonable(a.Get()) for a in prim.GetAttributes()
                 if a.GetName().startswith("inputs:") and a.HasAuthoredValue()}
        lights.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(), "inputs": attrs,
                       "visible": prim.GetAttribute("visibility").Get() if prim.HasAttribute("visibility") else None})
    return lights


def _materials(stage, roots):
    """Every material directly bound under the given roots, with its shader inputs and which prims use it."""
    from pxr import Usd, UsdShade

    materials = {}
    for root in roots:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim or not root_prim.IsValid():
            continue
        # TraverseInstanceProxies: the table and robot are instanceable USD references whose visuals (and material
        # bindings) live in prototypes, invisible to a plain traversal.
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                continue
            key = str(material.GetPath())
            if key not in materials:
                shaders = []
                for sprim in Usd.PrimRange(material.GetPrim(), Usd.TraverseInstanceProxies()):
                    if not sprim.IsA(UsdShade.Shader):
                        continue
                    shader = UsdShade.Shader(sprim)
                    source = shader.GetSourceAsset("mdl")
                    shaders.append({
                        "path": str(sprim.GetPath()),
                        "id": _jsonable(shader.GetShaderId()) or None,
                        "mdl_source": _jsonable(source) if source else None,
                        "inputs": {i.GetBaseName(): _jsonable(i.Get()) for i in shader.GetInputs() if i.Get() is not None},
                    })
                materials[key] = {"shaders": shaders, "bound_to": []}
            if len(materials[key]["bound_to"]) < 12:
                materials[key]["bound_to"].append(str(prim.GetPath()))
    return materials


def _events(env):
    events = {}
    cfg = getattr(env.cfg, "events", None)
    if cfg is None:
        return events
    for name, term in vars(cfg).items():
        func = getattr(term, "func", None)
        if func is None:
            continue
        events[name] = {"func": getattr(func, "__name__", str(func)), "mode": getattr(term, "mode", None),
                        "params": _jsonable({k: (v if isinstance(v, (int, float, str, bool, list, tuple, dict)) else str(v))
                                             for k, v in (term.params or {}).items()})}
    return events


def _reset_state(env):
    origin = env.scene.env_origins[0]
    state = {"rigid_objects": {}, "articulations": {}}
    for name, obj in env.scene.rigid_objects.items():
        state["rigid_objects"][name] = {"pos_in_env": _jsonable(obj.data.root_pos_w[0] - origin),
                                        "quat_wxyz": _jsonable(obj.data.root_quat_w[0])}
    for name, art in env.scene.articulations.items():
        state["articulations"][name] = {"joint_names": list(art.joint_names), "joint_pos": _jsonable(art.data.joint_pos[0])}
    return state


def _snapshot(env, stage):
    env_ns = env.scene.env_prim_paths[0]
    return {
        "lights": _lights(stage),
        "materials": _materials(stage, [env_ns, "/World/GroundPlane", "/World/ground"]),
        "state": _reset_state(env),
    }


def write_run_metadata(env, output_file, seed=None, task=None):
    """Write <output stem>_env.json and <output stem>_env_cfg.yaml next to the dataset file. Call once, after the
    first env.reset(). Returns the JSON path, or None on failure (never raises)."""
    try:
        import omni.usd
        from isaaclab.utils.io import dump_yaml

        out_dir, stem = Path(output_file).parent, Path(output_file).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / f"{stem}_env_cfg.yaml"
        dump_yaml(str(cfg_path), env.cfg)
        print(f"[env_metadata] wrote {cfg_path}")

        stage = omni.usd.get_context().get_stage()
        snap = _snapshot(env, stage)
        record = {
            "dataset_file": str(Path(output_file).resolve()),
            "task": task,
            "datagen_seed": seed,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sim": {"dt": env.cfg.sim.dt, "decimation": env.cfg.decimation, "render_interval": env.cfg.sim.render_interval,
                    "device": str(env.device), "num_envs": env.num_envs,
                    "num_rerenders_on_reset": getattr(env.cfg, "num_rerenders_on_reset", None),
                    "antialiasing_mode": getattr(getattr(env.cfg.sim, "render", None), "antialiasing_mode", None)},
            "eval_mode": getattr(env.cfg, "eval_mode", None),
            "eval_type": getattr(env.cfg, "eval_type", None),
            "note": ("lights/materials below are the live stage values after the first reset; with eval_mode off this "
                     "task's light/texture events reset them to these defaults every episode. Per-demo object layout is "
                     "in the dataset itself (each demo's initial_state)."),
            "cameras": _cameras(env),
            "lights": snap["lights"],
            "materials": snap["materials"],
            "events": _events(env),
            "first_reset_state": snap["state"],
        }
        out = out_dir / f"{stem}_env.json"
        out.write_text(json.dumps(record, indent=1, default=str))
        print(f"[env_metadata] wrote {out}")
        return out
    except Exception as e:  # metadata must never break data generation
        print(f"[env_metadata] could not write metadata: {type(e).__name__}: {e}")
        return None


def log_episodes(env, output_file):
    """One JSON line per episode written to the main dataset file, in <output stem>_episodes.jsonl.

    Snapshots cameras / lights / materials right after every env.reset(), and when the recorder exports an episode,
    appends that snapshot together with the file and group name the episode was written under (demo_N). Failed
    attempts routed to <stem>_failed.hdf5 are not logged. So line k is demo_k of the main file, whatever eval_mode
    randomized for it. Wraps env.reset and env.recorder_manager.export_episodes
    on this env instance only; Isaac Lab itself is not modified. Returns the JSONL path."""
    import omni.usd

    out_path = Path(output_file).parent / f"{Path(output_file).stem}_episodes.jsonl"
    rm = env.recorder_manager
    pending = {}  # env_id -> snapshot taken at that env's latest reset
    counter = {"reset": 0}

    original_reset = env.reset

    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        try:
            env_ids = kwargs.get("env_ids", args[1] if len(args) > 1 else None)
            ids = list(range(env.num_envs)) if env_ids is None else [int(i) for i in _jsonable(env_ids)]
            stage = omni.usd.get_context().get_stage()
            cameras = _cameras_all_envs(env, ids)
            lights, materials = _lights(stage), _materials(stage, [env.scene.env_prim_paths[0], "/World/GroundPlane"])
            for i in ids:
                pending[i] = {"reset_index": counter["reset"], "cameras": cameras[i], "lights": lights,
                              "materials": materials}
        except Exception as e:
            print(f"[env_metadata] snapshot after reset {counter['reset']} failed: {type(e).__name__}: {e}")
        counter["reset"] += 1
        return result

    original_export = rm.export_episodes

    def export_episodes(env_ids=None, demo_ids=None):
        # Predict where each episode goes, mirroring RecorderManager.export_episodes' routing, then confirm with
        # the handlers' demo counters after the real export.
        ids = list(range(env.num_envs)) if env_ids is None else [int(i) for i in _jsonable(env_ids)]
        demo_ids_list = None if demo_ids is None else [int(i) for i in _jsonable(demo_ids)]
        handlers = {"main": getattr(rm, "_dataset_file_handler", None), "failed": getattr(rm, "_failed_episode_dataset_file_handler", None)}
        before = {k: (h.demo_count if h is not None else None) for k, h in handlers.items()}
        planned = []
        next_index = dict(before)
        for n, env_id in enumerate(ids):
            episode = rm._episodes.get(env_id)
            if episode is None or episode.is_empty():
                continue
            success = bool(episode.success) if episode.success is not None else False
            # .name, not str(): DatasetExportMode is an IntEnum, and str() of an IntEnum is just the number in
            # Python 3.11+, which would silently route every episode to the main file here.
            mode = getattr(rm.cfg.dataset_export_mode, "name", str(rm.cfg.dataset_export_mode))
            if "SEPARATE_FILES" in mode:
                which = "main" if success else "failed"
            elif "SUCCEEDED_ONLY" in mode:
                which = "main" if success else None
            else:
                which = "main"
            if which is None or handlers[which] is None:
                continue
            name_index = demo_ids_list[n] if demo_ids_list is not None else next_index[which]
            if demo_ids_list is None:
                next_index[which] += 1
            num_samples = len(episode.data["actions"]) if "actions" in episode.data else None
            planned.append((env_id, which, f"demo_{name_index}", success, num_samples))

        result = original_export(env_ids, demo_ids)

        try:
            written = {k: (h.demo_count - before[k] if h is not None else 0) for k, h in handlers.items()}
            expected = {k: sum(1 for p in planned if p[1] == k) for k in handlers}
            if demo_ids_list is None and written != expected:
                print(f"[env_metadata] export routing mismatch (expected {expected}, written {written}) - "
                      "episode names in the log may be wrong for this export")
            with open(out_path, "a") as f:
                for env_id, which, name, success, num_samples in planned:
                    if which != "main":
                        continue  # only episodes in the main dataset file are logged (failed attempts are not)
                    h = handlers[which]
                    file_name = getattr(getattr(h, "_hdf5_file_stream", None), "filename", None)
                    record = {"dataset_file": os.path.basename(file_name) if file_name else which, "episode": name,
                              "success": success, "num_samples": num_samples, "env_id": env_id,
                              **pending.get(env_id, {"reset_index": None})}
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"[env_metadata] episode log failed: {type(e).__name__}: {e}")
        return result

    env.reset = reset
    rm.export_episodes = export_episodes
    print(f"[env_metadata] logging one line per exported episode to {out_path}")
    return out_path


def _cameras_all_envs(env, env_ids):
    """Per env: intrinsics, plus world pose for cameras fixed in the scene or the mount offset for cameras on the robot."""
    out = {i: {} for i in env_ids}
    for name, sensor in env.scene.sensors.items():
        data = getattr(sensor, "data", None)
        if data is None or getattr(data, "intrinsic_matrices", None) is None:
            continue
        mounted = _on_articulation(env, sensor)
        for i in env_ids:
            cam = {"intrinsic_matrix": _jsonable(data.intrinsic_matrices[i]),
                   "width": sensor.cfg.width, "height": sensor.cfg.height}
            if mounted:
                cam["mount_offset"] = {"parent": sensor.cfg.prim_path.rsplit("/", 1)[0], "pos": _jsonable(sensor.cfg.offset.pos),
                                       "rot_wxyz": _jsonable(sensor.cfg.offset.rot), "convention": sensor.cfg.offset.convention}
            else:
                cam["pos_in_env"] = _jsonable(data.pos_w[i] - env.scene.env_origins[i])
                cam["quat_w_ros_wxyz"] = _jsonable(data.quat_w_ros[i])
            out[i][name] = cam
    return out
