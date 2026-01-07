# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import pandas as pd

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 解析命令行参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# 是否录制视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# 视频录制时长（步数）
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# 是否禁用 Fabric，改用 USD I/O 操作
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# 模拟的环境数量
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# 运行的任务名称
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# 是否使用预训练的检查点
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# 是否以实时模式运行
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# 添加 RSL-RL 相关参数（如 PPO 训练参数）
cli_args.add_rsl_rl_args(parser)
# 添加 AppLauncher 相关参数（如 Isaac Sim 运行参数）
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
#  启用摄像头 进行录制
if args_cli.video:
    args_cli.enable_cameras = True

# 创建 AppLauncher 实例，启动 Omniverse（Isaac Sim 物理模拟器）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# 强化学习环境接口
import gymnasium as gym
import os
import time
import torch

# RSL-RL 的强化学习 PPO 训练器
from rsl_rl.runners import OnPolicyRunner
# 多智能体强化学习（MARL）
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# 导出模型为 JIT 或 ONNX 格式
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """加载 RSL-RL 训练好的智能体，在 Isaac Sim 进行推理"""
    # 解析任务（命令行参数）
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # 解析强化学习训练的超参数（如 PPO 配置）
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 选择检查点，在整个代码根目录下的，即IsaacLab/logs/rsl_rl/下
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # 使用 官方预训练模型
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # 使用 本地训练好的模型
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    # 读取 最新的实验检查点（默认）
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 创建 Gym 环境（可视化模式 rgb_array 用于录制视频）
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 转换为单智能体环境（如 RSL-RL 不支持多智能体）
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    # 创建环境
    # 如果开启视频录制，则包装环境，保存视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # 包装环境以适配 RSL-RL 算法
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # 加载训练好的 PPO 代理
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # 这行代码从指定路径 resume_path 加载之前训练好的PPO模型
    ppo_runner.load(resume_path)

    # 从PPO代理中提取出用于推理的策略
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 导出策略为ONNX/JIT格式
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # 定义了导出模型的存储路径，存储路径是在训练检查点的文件夹中创建一个 exported 目录
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    # 将PPO代理中的策略（actor_critic）以及观察归一化器（obs_normalizer）导出为JIT（Just-In-Time）格式，
    # 并保存为 policy.pt 文件
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    # 获取模拟环境的物理时间步长
    dt = env.unwrapped.physics_dt

    # 循环执行策略推理，控制智能体在 Isaac Sim 运行
    obs, _ = env.get_observations()
    # 计时器
    timestep = 0

    # 新增：初始化数据记录变量
    import isaaclab.utils.math as math_utils
    import math
    data_log = {
        'joint_pos': [],
        'base_euler': [],
        'platform_vel': []
    }

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():  # 记录当前时间，用于计算每一步的运行时间，确保实时仿真
            actions = policy(obs)  # 根据观测值 obs 计算动作
            obs, _, _, _ = env.step(actions)  # 计算环境的下一步状态
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:  # 当达到设定的 video_length 时，退出循环，停止录制
                break
        sleep_time = dt - (time.time() - start_time)  # 计算本次推理的运行时间并与物理时间步长 dt 比较

        # 新增：获取状态信息 -------------------------------------------------
        def euler_xyz_to_rot_mat(euler_angles: torch.Tensor) -> torch.Tensor:
            """将欧拉角 (XYZ 顺序) 转换为旋转矩阵。

            参数:
            euler_angles: 形状为 [..., 3] 的张量，表示欧拉角 (roll, pitch, yaw)。

            返回:
            形状为 [..., 3, 3] 的旋转矩阵。
            """
            # 确保 euler_angles 的形状为 [..., 3]
            if euler_angles.shape[-1] != 3:
                raise ValueError(f"Expected euler_angles to have shape [..., 3], but got {euler_angles.shape}")
            # 解包欧拉角
            roll = euler_angles[..., 0]
            pitch = euler_angles[..., 1]
            yaw = euler_angles[..., 2]
            # 计算旋转矩阵的各个分量
            cos_r, sin_r = torch.cos(roll), torch.sin(roll)
            cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
            cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
            # 构造旋转矩阵
            rot_mat = torch.stack([
                cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r,
                sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r,
                -sin_p, cos_p * sin_r, cos_p * cos_r
            ], dim=-1).reshape(*euler_angles.shape[:-1], 3, 3)
            return rot_mat

        # 获取机器人实例
        robot = env.unwrapped.scene["robot"]  # Articulation 对象
        # 获取platform刚体信息
        platform = env.unwrapped.scene["platform"]  # RigidObject 对象

        # # =========R_r^n      w_r/n^r      a_rn_r      alpha_rn_r=========
        # print(f"R_r^n : {robot.data.root_quat_w[0].cpu().numpy().round(3)}")
        # print(f"w_r/n^r: {robot.data.root_ang_vel_b[0].cpu().numpy().round(3)}")
        # a_rn_r = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], robot.data.body_lin_acc_w[0, 0])
        # print(f"a_rn_r: {a_rn_r.cpu().numpy().round(3)}")
        # alpha_rn_r = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], robot.data.body_ang_acc_w[0, 0])
        # print(f"alpha_rn_r: {alpha_rn_r.cpu().numpy().round(3)}")

        # # =========R_p^n      w_p/n^p      a_pn_p      alpha_pn_p=========
        # print(f"R_p^n: {platform.data.root_quat_w[0].cpu().numpy().round(3)}")
        # print(f"w_p/n^p: {platform.data.root_ang_vel_b[0].cpu().numpy().round(3)}")
        # a_pn_p = math_utils.quat_rotate_inverse(platform.data.root_quat_w[0], platform.data.body_lin_acc_w[0, 0])
        # print(f"a_pn_p: {a_pn_p.cpu().numpy().round(3)}")
        # alpha_pn_p = math_utils.quat_rotate_inverse(platform.data.root_quat_w[0], platform.data.body_ang_acc_w[0, 0])
        # print(f"alpha_pn_p: {alpha_pn_p.cpu().numpy().round(3)}")

        # # =========R_p_r      P_pr_r=========
        # R_p_r = math_utils.quat_mul(platform.data.root_com_quat_w[0], math_utils.quat_conjugate(robot.data.root_com_quat_w[0]))
        # print(f"R_p_r: {R_p_r.cpu().numpy().round(3)}")
        # P_pr_n = platform.data.root_pos_w[0] - robot.data.root_pos_w[0]
        # P_pr_r = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], P_pr_n)
        # print(f"P_pr_r: {P_pr_r.cpu().numpy().round(3)}")

        # # =========P_ir_r     V_ir_r     A_ir_r=========
        # P_ir_n = robot.data.body_pos_w[0,1] - robot.data.root_pos_w[0]
        # P_ir_r = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], P_ir_n)
        # print(f"P_ir_r: {P_ir_r.cpu().numpy().round(3)}")
        # V_ir_n = robot.data.body_lin_vel_w[0,1] - robot.data.root_lin_vel_w[0]
        # V_ir_r = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], V_ir_n)
        # print(f"V_ir_r: {V_ir_r.cpu().numpy().round(3)}")
        # A_ir_n = robot.data.body_lin_acc_w[0,1] - robot.data.body_lin_acc_w[0,0]
        # A_ir_r = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], A_ir_n)
        # print(f"A_ir_r: {A_ir_r.cpu().numpy().round(3)}")

        # torch.set_default_dtype(torch.float64)

        # def calculate_a_rn_r(input_values):

        #     x = input_values[0]
        #     y = input_values[1]

        #     # # 计算各项
        #     R_p_r = math_utils.quat_mul(platform.data.root_com_quat_w[0], math_utils.quat_conjugate(robot.data.root_com_quat_w[0]))
        #     a_pn_p = math_utils.quat_rotate_inverse(platform.data.root_com_quat_w[0], platform.data.body_lin_acc_w[0, 0])

        #     w_dot_rn_r = math_utils.quat_rotate_inverse(robot.data.root_com_quat_w[0], robot.data.body_ang_acc_w[0, 0])
        #     w_rn_r = math_utils.quat_rotate_inverse(robot.data.root_com_quat_w[0], robot.data.root_com_ang_vel_w[0])

        #     p_ir_r = math_utils.quat_rotate_inverse(robot.data.root_com_quat_w[0], robot.data.body_com_pos_w[0, y] - robot.data.root_com_pos_w[0])

        #     # 计算相对速度
        #     v_ir_n = robot.data.body_com_lin_vel_w[0, y] - robot.data.root_com_lin_vel_w[0]
        #     p_dot_ir_r = math_utils.quat_rotate_inverse(robot.data.root_com_quat_w[0], v_ir_n) - torch.cross(w_rn_r, p_ir_r)

        #     # 计算相对加速度
        #     a_ir_n = robot.data.body_lin_acc_w[0, y] - robot.data.body_lin_acc_w[0, 0]
        #     p_ddot_ir_r = (
        #         math_utils.quat_rotate_inverse(robot.data.root_com_quat_w[0], a_ir_n)
        #         - torch.cross(w_dot_rn_r, p_ir_r)
        #         - torch.cross(w_rn_r, torch.cross(w_rn_r, p_ir_r))
        #         - 2 * torch.cross(w_rn_r, p_dot_ir_r)
        #     )
        #     w_dot_pn_p = math_utils.quat_rotate_inverse(platform.data.root_com_quat_w[0], platform.data.body_ang_acc_w[0, 0])
        #     w_pn_p = math_utils.quat_rotate_inverse(platform.data.root_com_quat_w[0], platform.data.root_com_ang_vel_w[0])

        #     p_ip_p = math_utils.quat_rotate_inverse(platform.data.root_com_quat_w[0], robot.data.body_com_pos_w[0, y] - platform.data.root_com_pos_w[0])

        #     # 计算相对速度
        #     v_ip_n = robot.data.body_com_lin_vel_w[0, y] - platform.data.root_com_lin_vel_w[0]
        #     p_dot_ip_p = (
        #         math_utils.quat_rotate_inverse(platform.data.root_com_quat_w[0], v_ip_n)
        #         - torch.cross(w_pn_p, p_ip_p)
        #     )

        #     # 计算相对加速度
        #     a_ip_n = robot.data.body_lin_acc_w[0, y] - platform.data.body_lin_acc_w[0, 0]
        #     p_ddot_ip_p = (
        #         math_utils.quat_rotate_inverse(platform.data.root_com_quat_w[0], a_ip_n)
        #         - torch.cross(w_dot_pn_p, p_ip_p)
        #         - torch.cross(w_pn_p, torch.cross(w_pn_p, p_ip_p))
        #         - 2 * torch.cross(w_pn_p, p_dot_ip_p)
        #     )

        #     # 计算等式
        #     a_rn_r = (
        #         math_utils.quat_rotate(R_p_r, a_pn_p)
        #         - torch.cross(w_dot_rn_r, p_ir_r)
        #         - torch.cross(w_rn_r, torch.cross(w_rn_r, p_ir_r))
        #         - 2 * torch.cross(w_rn_r, p_dot_ir_r)
        #         - p_ddot_ir_r
        #         + math_utils.quat_rotate(R_p_r, torch.cross(w_dot_pn_p, p_ip_p) + torch.cross(w_pn_p, torch.cross(w_pn_p, p_ip_p)))
        #         + math_utils.quat_rotate(R_p_r, p_ddot_ip_p)
        #         + 2 * math_utils.quat_rotate(R_p_r, torch.cross(w_pn_p, p_dot_ip_p))
        #     )

        #     return a_rn_r

        def calculate_a_rn_r(input_values):
            x = input_values[0]
            y = input_values[1]

            def ensure_double(*args):
                return [arg.to(torch.float64) if isinstance(arg, torch.Tensor) else arg for arg in args]

            # ---- Step 1: 准备四元数与基本变量 ----
            quat_r, quat_p = ensure_double(robot.data.root_com_quat_w[0], platform.data.root_com_quat_w[0])
            R_p_r = math_utils.quat_mul(quat_p, math_utils.quat_conjugate(quat_r))

            a_pn_p = math_utils.quat_rotate_inverse(quat_p, platform.data.body_lin_acc_w[0, 0].double())

            w_dot_rn_r = math_utils.quat_rotate_inverse(quat_r, robot.data.body_ang_acc_w[0, 0].double())
            w_rn_r = math_utils.quat_rotate_inverse(quat_r, robot.data.root_com_ang_vel_w[0].double())

            p_ir_r = math_utils.quat_rotate_inverse(
                quat_r,
                (robot.data.body_com_pos_w[0, y] - robot.data.root_com_pos_w[0]).double()
            )

            v_ir_n = (robot.data.body_com_lin_vel_w[0, y] - robot.data.root_com_lin_vel_w[0]).double()
            p_dot_ir_r = math_utils.quat_rotate_inverse(quat_r, v_ir_n) - torch.cross(w_rn_r, p_ir_r)

            a_ir_n = (robot.data.body_lin_acc_w[0, y] - robot.data.body_lin_acc_w[0, 0]).double()
            p_ddot_ir_r = (
                    math_utils.quat_rotate_inverse(quat_r, a_ir_n)
                    - torch.cross(w_dot_rn_r, p_ir_r)
                    - torch.cross(w_rn_r, torch.cross(w_rn_r, p_ir_r))
                    - 2 * torch.cross(w_rn_r, p_dot_ir_r)
            )

            w_dot_pn_p = math_utils.quat_rotate_inverse(quat_p, platform.data.body_ang_acc_w[0, 0].double())
            w_pn_p = math_utils.quat_rotate_inverse(quat_p, platform.data.root_com_ang_vel_w[0].double())

            p_ip_p = math_utils.quat_rotate_inverse(
                quat_p,
                (robot.data.body_com_pos_w[0, y] - platform.data.root_com_pos_w[0]).double()
            )

            v_ip_n = (robot.data.body_com_lin_vel_w[0, y] - platform.data.root_com_lin_vel_w[0]).double()
            p_dot_ip_p = math_utils.quat_rotate_inverse(quat_p, v_ip_n) - torch.cross(w_pn_p, p_ip_p)

            a_ip_n = (robot.data.body_lin_acc_w[0, y] - platform.data.body_lin_acc_w[0, 0]).double()
            p_ddot_ip_p = (
                    math_utils.quat_rotate_inverse(quat_p, a_ip_n)
                    - torch.cross(w_dot_pn_p, p_ip_p)
                    - torch.cross(w_pn_p, torch.cross(w_pn_p, p_ip_p))
                    - 2 * torch.cross(w_pn_p, p_dot_ip_p)
            )

            # ---- Step 3: 计算最终表达式 ----
            a_rn_r = (
                    math_utils.quat_rotate(R_p_r, a_pn_p)
                    - torch.cross(w_dot_rn_r, p_ir_r)
                    - torch.cross(w_rn_r, torch.cross(w_rn_r, p_ir_r))
                    - 2 * torch.cross(w_rn_r, p_dot_ir_r)
                    - p_ddot_ir_r
                    + math_utils.quat_rotate(R_p_r, torch.cross(w_dot_pn_p, p_ip_p) + torch.cross(w_pn_p,
                                                                                                  torch.cross(w_pn_p,
                                                                                                              p_ip_p)))
                    + math_utils.quat_rotate(R_p_r, p_ddot_ip_p)
                    + 2 * math_utils.quat_rotate(R_p_r, torch.cross(w_pn_p, p_dot_ip_p))
            )

            return a_rn_r

        a_rn_r_true = math_utils.quat_rotate_inverse(robot.data.root_com_quat_w[0], robot.data.body_lin_acc_w[0, 0])
        print(f"a_rn_r_true: {a_rn_r_true.cpu().numpy().round(3)}")
        # print(f"a_rn_r0: {calculate_a_rn_r([0, 0]).cpu().numpy().round(3)}")
        # print(f"a_rn_r1: {calculate_a_rn_r([0, 1]).cpu().numpy().round(3)}")
        # print(f"a_rn_r6: {calculate_a_rn_r([0, 2]).cpu().numpy().round(3)}")
        # print(f"a_rn_r11: {calculate_a_rn_r([0, 3]).cpu().numpy().round(3)}")
        # print(f"a_rn_r15: {calculate_a_rn_r([0, 4]).cpu().numpy().round(3)}")
        # print(f"a_rn_rz: {calculate_a_rn_r([0, 5]).cpu().numpy().round(3)}")

        # =======================================================================================================
        # 创建保存目录
        # save_dir = "acc_data"
        # os.makedirs(save_dir, exist_ok=True)

        # # 定义保存函数
        # def save_tensor_row_to_excel(tensor, save_dir, filename, columns=None):
        #     # Ensure the directory exists
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)  # Create the directory if it doesn't exist

        #     # Create the full file path
        #     filepath = os.path.join(save_dir, filename)

        #     # Convert tensor to numpy array and reshape to 1 row (3 columns)
        #     array = tensor.detach().cpu().numpy().reshape(1, -1)  # Ensure it's a 1-row array
        #     df_new = pd.DataFrame(array, columns=columns)

        #     # Check if file exists
        #     if os.path.exists(filepath):
        #         # If it exists, append new data
        #         try:
        #             df_existing = pd.read_excel(filepath, engine="openpyxl")
        #             df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        #         except Exception as e:
        #             print(f"Error reading the existing file: {e}. Overwriting the file.")
        #             df_combined = df_new
        #     else:
        #         # If the file does not exist, create a new file with the first entry
        #         df_combined = df_new

        #     # Save the combined dataframe back to the Excel file
        #     try:
        #         with pd.ExcelWriter(filepath, engine="openpyxl", mode="w") as writer:
        #             df_combined.to_excel(writer, index=False)
        #         # print(f"Data saved successfully to {filepath}")
        #     except Exception as e:
        #         print(f"Error saving to {filepath}: {e}")
        # # 保存每个数组
        # save_tensor_row_to_excel(a_rn_r_true, save_dir,"a_rn_r_true.xlsx", columns=["x", "y", "z"])
        # save_tensor_row_to_excel(calculate_a_rn_r([0, 0]), save_dir, "a_rn_r0.xlsx", columns=["x", "y", "z"])
        # save_tensor_row_to_excel(calculate_a_rn_r([0, 1]), save_dir, "a_rn_r1.xlsx", columns=["x", "y", "z"])
        # save_tensor_row_to_excel(calculate_a_rn_r([0, 6]), save_dir, "a_rn_r2.xlsx", columns=["x", "y", "z"])
        # save_tensor_row_to_excel(calculate_a_rn_r([0, 11]), save_dir, "a_rn_r3.xlsx", columns=["x", "y", "z"])
        # save_tensor_row_to_excel(calculate_a_rn_r([0, 15]), save_dir, "a_rn_r4.xlsx", columns=["x", "y", "z"])

        # =======================================================================================================
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭模拟环境
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
