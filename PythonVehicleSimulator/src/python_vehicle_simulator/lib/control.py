#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control methods.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
from python_vehicle_simulator.lib.guidance import refModel3
from python_vehicle_simulator.lib.gnc import ssa, Rzyx

# SISO PID pole placement
def PIDpolePlacement(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    m,
    d,
    k,
    wn_d,
    zeta_d,
    wn,
    zeta,
    r,
    v_max,
    sampleTime,
):

    # PID gains based on pole placement
    Kp = m * wn ** 2.0 - k
    Kd = m * 2.0 * zeta * wn - d
    Ki = (wn / 10.0) * Kp

    # PID control law
    u = -Kp * e_x - Kd * e_v - Ki * e_int

    # Integral error, Euler's method
    e_int += sampleTime * e_x

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return u, e_int, x_d, v_d, a_d

#### 原始版本
#### MIMO nonlinear PID pole placement
# def DPpolePlacement(
#     e_int, M3, D3, eta3, nu3, x_d, y_d, psi_d, wn, zeta, eta_ref, sampleTime
# ):
#
#     # PID gains based on pole placement
#     M3_diag = np.diag(np.diag(M3))
#     D3_diag = np.diag(np.diag(D3))
#
#     Kp = wn @ wn @ M3_diag
#     Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
#     Ki = (1.0 / 10.0) * wn @ Kp
#
#     # DP control law - setpoint regulation
#     e = eta3 - np.array([x_d, y_d, psi_d])
#     e[2] = ssa(e[2])
#     R = Rzyx(0.0, 0.0, eta3[2])
#     tau = (
#         - np.matmul((R.T @ Kp), e)
#         - np.matmul(Kd, nu3)
#         - np.matmul((R.T @ Ki), e_int)
#     )
#
#     # Low-pass filters, Euler's method
#     T = 5.0 * np.array([1 / wn[0][0], 1 / wn[1][1], 1 / wn[2][2]])
#     x_d += sampleTime * (eta_ref[0] - x_d) / T[0]
#     y_d += sampleTime * (eta_ref[1] - y_d) / T[1]
#     psi_d += sampleTime * (eta_ref[2] - psi_d) / T[2]
#
#     # Integral error, Euler's method
#     e_int += sampleTime * e
#
#     return tau, e_int, x_d, y_d, psi_d


#### GMY change
def DPpolePlacement(
    e_int, M3, D3, eta3, nu3,
    eta_d, eta_dot, eta_ddot,  # ⬅ 新增：参考系统状态
    wn, zeta, eta_ref, sampleTime
):
    # PID gains based on pole placement
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))

    # 原始DP定义
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp


    # DP control law - setpoint regulation
    e = eta3 - eta_d
    # print("!!!error!!!:", e)
    e[2] = ssa(e[2])
    R = Rzyx(0.0, 0.0, eta3[2])
    tau = (
        - np.matmul((R.T @ Kp), e)
        - np.matmul(Kd, nu3)
        - np.matmul((R.T @ Ki), e_int)
    )


    # # Low-pass filters, Euler's method
    T = 5.0 * np.array([1 / wn[0][0], 1 / wn[1][1], 1 / wn[2][2]])
    eta_d, eta_dot, eta_ddot = update_reference_trajectory(
        eta_d, eta_dot, eta_ddot,
        eta_ref,  # 即 r_n
        sampleTime=sampleTime
    )

    # Integral error, Euler's method
    e_int += sampleTime * e

    return tau, e_int, eta_d, eta_dot, eta_ddot

##### GMY change2
# def DPpolePlacement(
#     M3, D3, eta3,
#     eta_d, eta_dot, eta_ddot,  # ⬅ 新增：参考系统状态
#     wn, zeta, eta_ref, sampleTime, tau3,
#     y_hat, xi_hat, b_hat, nu_hat, eta_hat
# ):
#     # 参数定义
#     T = np.diag([1000, 1000, 1000])
#     invT = np.linalg.inv(T)
#     # --- omega_o 和 omega_c ---
#     omega_o = 0.7 * np.ones(3)
#     omega_c = 1.2255 * omega_o
#     # --- 控制器增益参数 ---
#     zeta_ni = 1.0
#     lambda_ni = 0.1
#     # --- kesi ---
#     kesi = 0.1
#     # Reference system
#     OMEGA = np.diag(omega_o)
#     DELTA = np.diag([kesi, kesi, kesi])
#     # K1 系数（2行 × 3列块对角形式）
#     K_11 = -2 * (zeta_ni - lambda_ni) * omega_c[0] / omega_o[0]
#     K_12 = -2 * (zeta_ni - lambda_ni) * omega_c[1] / omega_o[1]
#     K_13 = -2 * (zeta_ni - lambda_ni) * omega_c[2] / omega_o[2]
#     K_14 = 2 * omega_o[0] * (zeta_ni - lambda_ni)
#     K_15 = 2 * omega_o[1] * (zeta_ni - lambda_ni)
#     K_16 = 2 * omega_o[2] * (zeta_ni - lambda_ni)
#     K1_upper = np.diag([K_11, K_12, K_13])
#     K1_lower = np.diag([K_14, K_15, K_16])
#     K1 = np.vstack([K1_upper, K1_lower])  # shape = (6,3)
#     # K2
#     K2 = np.diag([omega_c[0], omega_c[1], omega_c[2]])
#     # K4 和 K3
#     K4 = 1e5 * np.diag([1, 1, 1])
#     K3 = 0.1 * K4
#     # Aw（6x6）矩阵
#     Aw = np.block([
#         [np.zeros((3, 3)), np.eye(3)],
#         [-OMEGA @ OMEGA, -2 * DELTA @ OMEGA]
#     ])
#     # Cw（3x6）矩阵：提取后3列速度项
#     Cw = np.hstack([np.zeros((3, 3)), np.eye(3)])
#     # controller
#     Kp = 1e5 * np.diag([4e2, 1e0, 1e4])
#     Kd = 0e0 * np.diag([1e1, 1e1, 1e0])
#
#     R = Rzyx(0.0, 0.0, eta3[2])
#
#     # 给定一些初始值（确保是否要转化为列向量相乘）
#     tau_wind = np.zeros(3)   # tau_wind 不确定是否要加
#
#
#     # PID gains based on pole placement
#     M3_diag = np.diag(np.diag(M3))
#     D3_diag = np.diag(np.diag(D3))
#     Minv = np.linalg.inv(M3_diag)
#
#     y = eta3 # + np.random.normal(loc=0.0, scale=0.05, size=eta3.shape)
#     y_tilde = y - y_hat
#
#     xi_hat_dot = Aw @ xi_hat + K1 @ y_tilde
#     eta_hat_dot = R @ nu_hat + K2 @ y_tilde
#     b_hat_dot = -invT @ b_hat + K3 @ y_tilde
#     nu_hat_dot = np.matmul(Minv, -D3_diag @ nu_hat + R.T @ b_hat + tau3 + tau_wind + R.T @ K4 @ y_tilde)
#
#     xi_hat = xi_hat + sampleTime * xi_hat_dot
#     b_hat = b_hat + sampleTime * b_hat_dot
#     eta_hat = eta_hat + sampleTime * eta_hat_dot
#     nu_hat = nu_hat + sampleTime * nu_hat_dot
#     y_hat = eta_hat + Cw @ xi_hat
#
#
#     # # # 原始DP定义
#     # Kp = wn @ wn @ M3_diag
#     # Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
#     # Ki = (1.0 / 10.0) * np.diag(wn) @ Kp  # GMY change ： np.diag(wn)替换了wn
#
#
#     # e = eta3 - eta_d
#     # # print("!!!error!!!:", e)
#     # e[2] = ssa(e[2])
#
#     eta_d, eta_dot, eta_ddot = update_reference_trajectory(
#         eta_d, eta_dot, eta_ddot,
#         eta_ref,  # 即 r_n
#         sampleTime=sampleTime
#     )
#
#     tau3 = (
#             - tau_wind
#             - np.matmul((R.T @ Kp), (eta_hat - eta_d))
#             - np.matmul((R.T @ Kd @ R), nu_hat)
#             - np.matmul(R.T, b_hat)
#     )
#
#
#     return tau3, eta_d, eta_dot, eta_ddot, y_hat, xi_hat, b_hat, nu_hat, eta_hat


# GMY change
def update_reference_trajectory(
    eta_d, eta_dot, eta_ddot, r_n, sampleTime
):
    """
    单步更新三阶参考轨迹系统
    eta_d, eta_dot, eta_ddot: 当前状态
    r_n: 最终目标位置
    sampleTime: 时间步长
    """
    omega_o = np.array([0.1, 0.1, 0.2])
    kesi = 0.1
    OMEGA = np.diag(omega_o)
    DELTA = np.diag([kesi, kesi, kesi])
    I3 = np.eye(3)

    Omega2 = OMEGA @ OMEGA
    Omega3 = Omega2 @ OMEGA
    coeff = (2 * DELTA + I3)

    a2 = coeff @ OMEGA
    a1 = coeff @ Omega2
    a0 = Omega3

    # 三阶导数（jerk）
    eta_ddd = - a2 @ eta_ddot - a1 @ eta_dot - a0 @ eta_d + a0 @ r_n

    # Euler 积分
    eta_ddot_new = eta_ddot + sampleTime * eta_ddd
    eta_dot_new  = eta_dot + sampleTime * eta_ddot
    eta_d_new    = eta_d + sampleTime * eta_dot
    # print("[INFO] 参考轨迹:", eta_d_new)
    # print("[INFO] 参考轨迹d:", eta_dot_new)
    # print("[INFO] 参考轨迹dd:", eta_ddot_new)
    return eta_d_new, eta_dot_new, eta_ddot_new

# Heading autopilot - Intergral SMC (Equation 16.479 in Fossen 2021)
def integralSMC(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    T_nomoto,
    K_nomoto,
    wn_d,
    zeta_d,
    K_d,
    K_sigma,
    lam,
    phi_b,
    r,
    v_max,
    sampleTime,
):

    # Sliding surface
    v_r_dot = a_d - 2 * lam * e_v - lam ** 2 * ssa(e_x)
    v_r     = v_d - 2 * lam * ssa(e_x) - lam ** 2 * e_int
    sigma   = e_v + 2 * lam * ssa(e_x) + lam ** 2 * e_int

    #  Control law
    if abs(sigma / phi_b) > 1.0:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * np.sign(sigma) ) / K_nomoto
    else:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * (sigma / phi_b) ) / K_nomoto

    # Integral error, Euler's method
    e_int += sampleTime * ssa(e_x)

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return delta, e_int, x_d, v_d, a_d




