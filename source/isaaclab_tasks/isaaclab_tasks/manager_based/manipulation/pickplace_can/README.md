# `pickplace_can` — robosuite `PickPlaceCan` for Isaac Lab

An Isaac Lab port of the task behind robomimic's `can` datasets, on a Franka Panda.

## What is actually being ported

The stack people usually mean by "the robomimic can task" is three layers:

| Layer | Role |
| --- | --- |
| **MuJoCo** | physics engine **and** renderer |
| **robosuite** (1.5.2) | MJCF scene + task logic — `PickPlaceCan` lives here |
| **robomimic** | datasets (`can/ph`, `can/mh`) + BC/BC-RNN training |

robomimic does **not** simulate anything. So this port targets robosuite's
`environments/manipulation/pick_place.py::PickPlaceCan` and its `BinsArena`.

## How the can mesh is handled

robosuite renders the can straight from MJCF — it is a **956-triangle mesh**, not a primitive:

```xml
<mesh file="meshes/can.msh" name="can_mesh"/>
<texture file="../textures/soda.png" name="tex-can" type="2d"/>
<material name="coke" reflectance="0.7" texrepeat="5 5" texture="tex-can" texuniform="true"/>
<geom mesh="can_mesh" type="mesh" density="100" friction="0.95 0.3 0.1" material="coke" condim="4" .../>
```

MuJoCo uses that mesh for rendering and convex-hulls it for contact. Isaac Lab needs USD, so the
same `can.obj` (+ `can.mtl` + `soda.png`) is converted **once, offline**:

```bash
./isaaclab.sh -p scripts/tools/convert_robosuite_can.py
```

This wraps `scripts/tools/convert_mesh.py` (which drives `omni.kit.asset_converter`) and writes
`<IsaacLab>/assets/robosuite/can.usd`, copying `soda.png` next to it so the texture survives.
Override the location with `ROBOSUITE_CAN_USD=/path/to/can.usd`.

Collision defaults to `convexHull` because that is what MuJoCo effectively does with a mesh geom.
Use `--collision-approximation convexDecomposition` if you want the concave rim modelled.

If the USD is missing the task **falls back to a cylinder primitive** of identical bounding size
(r=0.025, h=0.0806) and logs a warning. Fine for low-dim work, useless for visuomotor — it has no
texture.

Measured from the source mesh:

| Property | Value | Source |
| --- | --- | --- |
| bounding box | 0.050 × 0.050 × 0.0806 m | `can.obj` vertices |
| volume | 1.4646e-4 m³ | divergence theorem over the triangles |
| mass | **14.65 g** | `density="100"` × volume |
| friction | 0.95 sliding | `friction="0.95 0.3 0.1"` (PhysX models only the sliding term) |
| origin offset | base is 0.0403 m below the mesh origin | not centred — matters for rest height |

## Frame convention

robosuite works in a floor-origin world frame. Isaac Lab manipulation tasks put the robot base at
the environment origin, and the mimic/IK machinery assumes that. Everything is therefore shifted by

```
env_local = robosuite_world - (-0.5, -0.1, 0.912)
```

* `(-0.5, -0.1)` = `PandaRobot.base_xpos_offset["bins"]`
* `0.912` = where the Panda root ends up on the Rethink pedestal
  (`bottom_offset - top_offset = -0.922 - (-0.01)`)

It is a pure translation, so all relative geometry is preserved exactly. Resulting layout:

| Thing | env-local |
| --- | --- |
| source bin (bin1) centre | (0.600, -0.150, -0.112) |
| target bin (bin2) centre | (0.600,  0.380, -0.112) |
| bin top surface | z = -0.092 |
| can rest height | z = -0.0517 |
| ground plane | z = -0.912 |

Note the bin surface is 0.092 m **below** the robot base — unlike the `stack` task where the table
top is level with it. That offset is real and comes from robosuite.

## ⚠️ Reachability: read this before collecting demos

robosuite assigns the can **object id 3**, which `not_in_bin` maps to the **+x/+y quadrant** of
bin2 — the corner furthest from the robot. Solving IK against the Panda's modified-DH model at
drop height:

| Point in the can's target quadrant | radial | IK |
| --- | --- | --- |
| inner corner (0.600, 0.380) | 0.710 m | ✅ solves |
| (0.660, 0.430) | 0.788 m | ✅ solves |
| quadrant centre (0.698, 0.502) | 0.860 m | ❌ 5.8 cm short |
| far corner (0.795, 0.625) | 1.011 m | ❌ 56 cm short |

Every can **spawn** pose in bin1 is comfortably reachable (0.58–0.78 m). It is only the *drop* that
is tight. The task is still solvable as shipped — success only needs the can's origin anywhere in
the quadrant, and the inner corner reaches — but the usable drop zone is a narrow wedge, which
makes teleoperation and mimic data generation fiddly.

This is inherent to robosuite's layout with a Panda, not a porting error.

If you would rather not fight it, use the **`NearBin`** variants. They reassign the can to
quadrant 0 (the one robosuite gives the milk carton), spanning x ∈ (0.405, 0.600),
y ∈ (0.135, 0.380) — every corner solves, furthest 0.710 m. Same arena, same can, same dynamics;
only the goal region moves. It is a deliberate deviation from robosuite, so demos collected there
are **not** geometrically comparable to robomimic's `can` datasets.

## Registered environments

| Gym id | Control | Target quadrant |
| --- | --- | --- |
| `Isaac-PickPlace-Can-Franka-v0` | joint position | faithful (id 3) |
| `Isaac-PickPlace-Can-Franka-IK-Rel-v0` | **relative IK** ← closest to robosuite OSC_POSE | faithful (id 3) |
| `Isaac-PickPlace-Can-Franka-IK-Abs-v0` | absolute IK | faithful (id 3) |
| `Isaac-PickPlace-Can-NearBin-Franka-v0` | joint position | near (id 0) |
| `Isaac-PickPlace-Can-NearBin-Franka-IK-Rel-v0` | relative IK | near (id 0) |

robosuite drives `PickPlaceCan` with `OSC_POSE` (6 delta-pose + 1 gripper), so **`IK-Rel` is the
one to use** if you care about matching how the `can` demos were collected.

## Observations

Term names match the `stack` task so the same robomimic configs and `isaaclab_mimic` plumbing
apply unchanged.

| Group / term | Shape |
| --- | --- |
| `policy/actions` | (7,) |
| `policy/joint_pos`, `joint_vel` | (9,) |
| `policy/object` | (23,) |
| `policy/can_position` / `can_orientation` | (3,) / (4,) |
| `policy/target_position` | (3,) |
| `policy/eef_pos` / `eef_quat` | (3,) / (4,) |
| `policy/gripper_pos` | (2,) |
| `subtask_terms/grasp_can`, `lift_can` | (,) |

`object` is: can pos, can quat, gripper→can, can→target, target pos, gripper pos, gripper quat.
The `can→target` and `target` terms are additions — robosuite's `object-state` has no goal term
because the goal is baked into the env.

## Success

Transcribed from `PickPlace._check_success` for the can, which needs **both**:

1. the can's origin inside its quadrant box (x, y, and `bin2_z < z < bin2_z + 0.1`);
2. `r_reach < 0.6` where `r_reach = 1 - tanh(10·d)` — i.e. the gripper is **≥ 0.0424 m away**, so
   the can has actually been released rather than dangled over the bin.

Dropping condition 2 would let a policy "succeed" while still holding the can.

## Reset randomisation

Matches robosuite's `UniformRandomSampler` over bin1: uniform x ∈ (0.490, 0.710),
y ∈ (-0.310, 0.010), full random yaw. The box is pre-shrunk by the can's horizontal radius
(0.0354), which is how robosuite's `ensure_object_boundary_in_range=True` behaves.

## Deliberate omissions

* **Shaped reward.** robosuite defines a reach/grasp/lift/hover reward for `PickPlace`. Not ported —
  the robomimic `can` datasets are behaviour cloning and never use it. `rewards = None`.
* **Bin textures.** robosuite uses light-wood / dark-wood PNGs; this uses flat colours in the same
  family. Only the can's texture was worth carrying across.
* **Camera observations.** `RGBCameraPolicyCfg` is declared but empty. Add cameras the way
  `stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py` does.
* **The robot pedestal.** robosuite models a Rethink pedestal below the arm; the base offset it
  produces is applied, but the pedestal geometry is not drawn.

## Package name

It is `pickplace_can`, **not** `pick_place_can`. `isaaclab_tasks/__init__.py` blacklists the
substring `"pick_place"` from auto-registration and the match is a substring test, so a package
named `pick_place_can` would silently never register.
