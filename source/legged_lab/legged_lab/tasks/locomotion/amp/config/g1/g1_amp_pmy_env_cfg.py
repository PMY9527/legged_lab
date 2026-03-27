"""G1 AMP environment config with 2 LaFan1 clips (walk + run).

Designed for learning omnidirectional locomotion with just 2 reference clips.
Extends standard G1 AMP rewards with denser locomotion signals.
"""

import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

from .g1_amp_env_cfg import G1AmpEnvCfg, G1AmpRewards


@configclass
class G1AmpPMYRewards(G1AmpRewards):
    """
    is_alive, stand_still_joint_deviation, feet_contact_without_cmd
    """

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )

    stand_still_joint_deviation = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )


@configclass
class G1AmpPMYEnvCfg(G1AmpEnvCfg):
    """G1 AMP environment with 2 LaFan1 clips for walk/run."""

    rewards: G1AmpPMYRewards = G1AmpPMYRewards()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # motion data — only 2 clips (walk + run)
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "pmy"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "walk1_subject1": 1.0,
            "run1_subject2": 1.0,
        }


@configclass
class G1AmpPMYEnvCfg_PLAY(G1AmpPMYEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
