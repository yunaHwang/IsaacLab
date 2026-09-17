"""SO-101 stack-2-cubes scene rebuilt to match SCRAPE-IsaacLab's `Isaac-Stack2Cubes-SO101-Single-v0`
(the private task behind CoRL2026-CSI/Isaaclab-so101_11task_baseCaP_3300epi, episodes 600-899),
written against plain Isaac Lab so it runs in yuna_env without leisaac installed, and built
directly from the config class - no gym.register():

    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=SO101Stack2CubesEnvCfg())

WHAT IS MATCHED, AND WHERE EACH NUMBER CAME FROM (dataset meta + episode 600 frames)
    robot       leisaac's so101_follower.usd (LightwheelAI/leisaac release v0.1.0, saved under
                ./assets/robots/). The dataset's rest pose in radians (-2, -102, 87, 69, 0, 0 deg)
                is leisaac's SO-101 rest pose, so SCRAPE used the same asset/joint convention.
    units       dataset observation.state/action are LeRobot normalized motor units. The map to
                sim joint radians is exactly linear per joint (fit over all 300 episodes against
                the dataset's own observation.state.radian_urdf0, residual ~1e-7):
                    rad = NORM_TO_RAD_SCALE * u + NORM_TO_RAD_OFFSET
                leisaac's convert_lerobot_action_to_leisaac() uses a DIFFERENT calibration and
                would put every joint in the wrong place - do not use it for this checkpoint.
    cubes       4 cm (grasp z ~0.02, release z ~0.064), both placed in x [0.05, 0.40],
                y [-0.30, 0.30] of the robot base frame, >= 8 cm apart (from grasp/release EE
                positions of all 300 episodes). cube_top is always SCRAPE's `cube_2`.
    colors      the 11-color palette SCRAPE sampled per episode (scrape_episode_metadata.jsonl);
                the instruction names them: "Stack the {top} block on the {base} block."
    light       dome light, base intensity 2500, scaled uniformly in [0.25, 1.65] per episode.
    top camera  straight down, ~720 px/m at table height, image center over robot-frame
                (0.37, 0.007), image right = +x, image up = +y. Estimated from cube pixel
                centers vs grasp/release EE positions in episode 600 - approximate, compare
                renders with the dataset frames before trusting it.
    wrist cam   leisaac's wrist camera mount (the SCRAPE wrist view shows the same jaw framing).
    table       light wood top at z=0 spanning robot-frame x [-0.04, 0.79], white floor beyond.
    base pose   SCRAPE's robot frame (observation.ee_pos.robot_xyzrpy, cube labels) is the base_link of
                LeRobot's so101_new_calib.urdf (TheRobotStudio/SO-ARM100, Simulation/SO101): FK of that
                URDF at gripper_frame_link on observation.state.radian_urdf0 reproduces ee_pos to 0.000 mm.
                leisaac's USD has identical kinematics (0.00 mm residual, no joint zero offsets, episode
                674) but its root is NOT at that base_link: base_link sits (-0.0158, +0.0208, +0.0325) m
                from the USD root. So the root is spawned at minus that offset, which makes the env origin
                SCRAPE's robot frame exactly. Without it the whole arm is 3.25 cm too high (and ~2 cm off
                in xy) relative to the cubes and table.

NOT matched (unknown without SCRAPE's code): exact table/wood texture, light type and placement,
camera intrinsics beyond the px/m estimate, cube friction/mass, and the SCRAPE VLM success judge
(replaced by a geometric stacked check below).
"""

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SO101_USD = ASSETS_DIR / "robots" / "so101_follower.usd"

# Order used by the dataset's observation.state / action and by this env's action vector.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

NORM_TO_RAD_SCALE = [0.016030099, 0.018093303, 0.016958158, 0.017763498, 0.031408257, 0.021859226]
NORM_TO_RAD_OFFSET = [0.041417481, -0.0069029135, -0.17410682, -0.082834963, -0.00076699035, -0.0015339812]

# URDF base_link origin relative to the leisaac USD root, in world axes (with the +90 deg yaw below). The
# robot root is spawned at the negative of this so base_link lands on the env origin - see "base pose" above.
URDF_BASE_LINK_IN_USD_ROOT = (-0.015761, 0.020791, 0.032481)
ROBOT_ROOT_POS = tuple(-v for v in URDF_BASE_LINK_IN_USD_ROOT)

# SCRAPE's EE point (URDF gripper_frame_link) expressed in the USD "gripper" body frame. The URDF offset is
# (-0.0079, -0.000218, -0.098127) in gripper_link; the USD gripper body is gripper_link twisted 2.789 deg
# about z (the URDF wrist_roll origin's 0.0487 rad), which maps it to this. EE = gripper pos + R_gripper @ this.
GRIPPER_FRAME_OFFSET = (-0.007901, 0.000167, -0.098127)

# Episode 600 frame 0 (identical across the stack episodes checked), radians.
REST_JOINT_POS = [-0.0387995, -1.7726202, 1.5203599, 1.2011346, 0.0, -0.0006078]

# SCRAPE's SO-101 moves slightly past leisaac's USD joint limits: over all 3300 episodes the dataset
# reaches shoulder_lift -1.816 rad (USD: +-1.745) and wrist_flex -1.764 / +1.687 rad (USD: +-1.658).
# Isaac Lab rejects a default pose outside the USD limits at construction, so the config spawns with
# the rest pose clamped inside them, and widen_joint_limits_to_dataset() opens these limits after
# the env exists and restores the exact dataset rest pose.
DATASET_JOINT_LIMITS = {"shoulder_lift": (-1.85, 1.745), "wrist_flex": (-1.80, 1.72)}
_USD_LIMITS = {"shoulder_lift": (-1.745, 1.745), "wrist_flex": (-1.658, 1.658)}
SPAWN_JOINT_POS = [
    min(max(q, _USD_LIMITS[n][0] + 0.005), _USD_LIMITS[n][1] - 0.005) if n in _USD_LIMITS else q
    for n, q in zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"], REST_JOINT_POS)
]

CUBE_SIZE = 0.04
CUBE_X_RANGE = (0.05, 0.40)
CUBE_Y_RANGE = (-0.30, 0.30)
CUBE_MIN_SEPARATION = 0.08

# SCRAPE's per-episode palette (name -> linear rgb), from scrape_episode_metadata.jsonl.
CUBE_COLORS = {
    "pink": (0.9131, 0.1714, 0.5149),
    "gray": (0.2051, 0.2051, 0.2051),
    "cyan": (0.0015, 0.3515, 0.4452),
    "purple": (0.2086, 0.0497, 0.4621),
    "lime": (0.305, 0.7913, 0.013),
    "green": (0.1122, 0.4964, 0.2379),
    "red": (0.2836, 0.0194, 0.0194),
    "blue": (0.0003, 0.0497, 0.3814),
    "black": (0.03, 0.03, 0.03),
    "yellow": (0.6038, 0.5333, 0.0089),
    "orange": (0.6038, 0.2423, 0.0089),
}
LIGHT_BASE_INTENSITY = 2500.0
LIGHT_SCALE_RANGE = (0.25, 1.65)

TOP_CAM_HEIGHT = 1.0
TOP_CAM_XY = (0.37, 0.007)
TOP_CAM_PX_PER_M = 720.0  # at z ~0.03 (cube tops / grasp height)
_H_APERTURE = 20.955
_TOP_FOCAL = TOP_CAM_PX_PER_M * (TOP_CAM_HEIGHT - 0.03) * _H_APERTURE / 640.0


def normalized_to_radians(u: torch.Tensor) -> torch.Tensor:
    """Dataset/Pi0.5 normalized motor units [..., 6] -> sim joint radians [..., 6]."""
    scale = torch.tensor(NORM_TO_RAD_SCALE, dtype=u.dtype, device=u.device)
    offset = torch.tensor(NORM_TO_RAD_OFFSET, dtype=u.dtype, device=u.device)
    return u * scale + offset


def radians_to_normalized(q: torch.Tensor) -> torch.Tensor:
    """Sim joint radians [..., 6] -> dataset/Pi0.5 normalized motor units [..., 6]."""
    scale = torch.tensor(NORM_TO_RAD_SCALE, dtype=q.dtype, device=q.device)
    offset = torch.tensor(NORM_TO_RAD_OFFSET, dtype=q.dtype, device=q.device)
    return (q - offset) / scale


def _cube_cfg(name: str, pos, color) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class SO101Stack2CubesSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SO101_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POS,  # puts SCRAPE's URDF base_link at the env origin - see URDF_BASE_LINK_IN_USD_ROOT
            # +90 deg yaw (wxyz): the leisaac USD base frame is rotated relative to SCRAPE's robot frame.
            # Replaying episode 674 with identity rotation, the dataset EE path matched the sim gripper
            # path only after a 89.6-90.0 deg yaw (diagnose_so101_replay.py). With this rotation the
            # world frame IS SCRAPE's robot frame, which is where cubes and the top camera are placed.
            rot=(0.70710678, 0.0, 0.0, 0.70710678),
            joint_pos=dict(zip(JOINT_NAMES, SPAWN_JOINT_POS)),  # see DATASET_JOINT_LIMITS
        ),
        # Same actuator model as leisaac's SO101_FOLLOWER_CFG.
        actuators={
            "sts3215-gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"], effort_limit_sim=10, velocity_limit_sim=10, stiffness=17.8, damping=0.60
            ),
            "sts3215-arm": ImplicitActuatorCfg(
                joint_names_expr=JOINT_NAMES[:5], effort_limit_sim=10, velocity_limit_sim=10, stiffness=17.8, damping=0.60
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.83, 1.4, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.62, 0.42), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.375, 0.0, -0.02)),
    )

    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.GroundPlaneCfg(color=(0.9, 0.9, 0.9)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=LIGHT_BASE_INTENSITY),
    )

    cube_top: RigidObjectCfg = _cube_cfg("CubeTop", (0.34, 0.08, CUBE_SIZE / 2), CUBE_COLORS["gray"])
    cube_base: RigidObjectCfg = _cube_cfg("CubeBase", (0.24, -0.02, CUBE_SIZE / 2), CUBE_COLORS["pink"])

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        # ros convention (x right, y down, looking along +z): 180 deg about world x, so the
        # camera looks straight down with image right = +x and image up = +y.
        offset=TiledCameraCfg.OffsetCfg(pos=(*TOP_CAM_XY, TOP_CAM_HEIGHT), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOP_FOCAL, focus_distance=400.0, horizontal_aperture=_H_APERTURE, clipping_range=(0.01, 50.0)
        ),
        width=640,
        height=480,
    )

    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5, focus_distance=400.0, horizontal_aperture=36.83, clipping_range=(0.01, 50.0)
        ),
        width=640,
        height=480,
    )


@configclass
class ActionsCfg:
    # Absolute joint-position targets in radians, dataset joint order.
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES[:5], scale=1.0, use_default_offset=False, preserve_order=True
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)}
        )
        top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False})
        left_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False}
        )
        cube_top_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_top")})
        cube_base_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_base")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def reset_cubes_apart(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Place both cubes uniformly in the SCRAPE workspace, at least CUBE_MIN_SEPARATION apart, random yaw."""
    top: RigidObject = env.scene["cube_top"]
    base: RigidObject = env.scene["cube_base"]
    device = env.device
    for env_id in env_ids.tolist():
        while True:
            xy = torch.stack([
                torch.empty(2, device=device).uniform_(*CUBE_X_RANGE),
                torch.empty(2, device=device).uniform_(*CUBE_Y_RANGE),
            ], dim=-1)
            if torch.linalg.norm(xy[0] - xy[1]) >= CUBE_MIN_SEPARATION:
                break
        for obj, p in ((top, xy[0]), (base, xy[1])):
            yaw = torch.empty(1, device=device).uniform_(-torch.pi / 4, torch.pi / 4)
            pose = torch.zeros(1, 7, device=device)
            pose[0, :2] = p + env.scene.env_origins[env_id, :2]
            pose[0, 2] = CUBE_SIZE / 2 + env.scene.env_origins[env_id, 2]
            pose[0, 3] = torch.cos(yaw / 2)
            pose[0, 6] = torch.sin(yaw / 2)
            ids = torch.tensor([env_id], device=device)
            obj.write_root_pose_to_sim(pose, env_ids=ids)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=ids)


def cubes_stacked(env: ManagerBasedRLEnv, xy_tol: float = 0.02, z_tol: float = 0.012) -> torch.Tensor:
    """Top cube resting on the base cube: centered within xy_tol and one cube-height above it."""
    top = env.scene["cube_top"].data.root_pos_w
    base = env.scene["cube_base"].data.root_pos_w
    xy_ok = torch.linalg.norm(top[:, :2] - base[:, :2], dim=-1) < xy_tol
    z_ok = torch.abs((top[:, 2] - base[:, 2]) - CUBE_SIZE) < z_tol
    return xy_ok & z_ok


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cubes = EventTerm(func=reset_cubes_apart, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=cubes_stacked)


@configclass
class RewardsCfg:
    pass


@configclass
class SO101Stack2CubesEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101Stack2CubesSceneCfg = SO101Stack2CubesSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # 30 Hz control (the dataset's fps): physics at 60 Hz, 2 physics steps per env.step().
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 40.0  # stack episodes are ~35 s (693-1109 frames)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.eye = (-0.35, -0.55, 0.55)
        self.viewer.lookat = (0.3, 0.0, 0.0)


def widen_joint_limits_to_dataset(env: ManagerBasedRLEnv) -> None:
    """Open shoulder_lift / wrist_flex limits to the dataset's range (DATASET_JOINT_LIMITS) and make the
    exact dataset rest pose the reset pose. Call once, right after ManagerBasedRLEnv(cfg=...)."""
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(n) for n in DATASET_JOINT_LIMITS]
    limits = torch.tensor([DATASET_JOINT_LIMITS[n] for n in DATASET_JOINT_LIMITS], device=env.device)
    robot.write_joint_position_limit_to_sim(limits.unsqueeze(0).expand(env.num_envs, -1, -1), joint_ids=ids)
    order = [robot.joint_names.index(n) for n in JOINT_NAMES]
    rest = torch.tensor(REST_JOINT_POS, device=env.device)
    robot.data.default_joint_pos[:, order] = rest
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos))


# --- per-episode appearance (colors + light), applied to USD after env creation/reset ---------

def _set_diffuse_color(prim_path: str, rgb) -> None:
    from pxr import Gf, Sdf, Usd, UsdShade

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if prim.IsA(UsdShade.Shader):
            UsdShade.Shader(prim).CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))


def _set_light_intensity(prim_path: str, intensity: float) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.GetPrimAtPath(prim_path).GetAttribute("inputs:intensity").Set(float(intensity))


def randomize_appearance(env: ManagerBasedRLEnv, generator: torch.Generator | None = None, top_color=None, base_color=None,
                         light_scale=None) -> dict:
    """Pick SCRAPE-style cube colors + light scale for env 0, apply them, and return the matching
    instruction. Any argument left as None is sampled."""
    names = list(CUBE_COLORS)
    if top_color is None or base_color is None:
        perm = torch.randperm(len(names), generator=generator)
        top_color = top_color or names[perm[0]]
        base_color = base_color or next(n for n in (names[i] for i in perm.tolist()) if n != top_color)
    if light_scale is None:
        lo, hi = LIGHT_SCALE_RANGE
        light_scale = lo + (hi - lo) * torch.rand(1, generator=generator).item()

    env_ns = env.scene.env_prim_paths[0]
    _set_diffuse_color(f"{env_ns}/CubeTop", CUBE_COLORS[top_color])
    _set_diffuse_color(f"{env_ns}/CubeBase", CUBE_COLORS[base_color])
    _set_light_intensity("/World/Light", LIGHT_BASE_INTENSITY * light_scale)
    return {
        "top_color": top_color,
        "base_color": base_color,
        "light_scale": light_scale,
        "instruction": f"Stack the {top_color} block on the {base_color} block.",
    }
