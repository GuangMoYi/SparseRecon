#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   semisub.py 
        Class for a semisubmersible with mass m =  27 162 500 (mass), pontoon  
        Height H_p = 13.0 m, pontoon width B_p = 12.0 m, pontoon length
        L_p = 84.6 m. 

    semisub()                                      
        Step inputs for propeller revolutions n1, n2, n3 and n4
    semisub('DPcontrol',x_d,y_d,psi_d,V_c,beta_c) DP control system
        x_d: desired x position (m)
        y_d: desired y position (m)
        psi_d: desired yaw angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        
Methods:
    
[nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime)
    returns nu[k+1] and u_actual[k+1] using Euler's method. 
    The control inputs are:

    u_control = n (RPM)
    n = [ #1 Bow tunnel thruster (RPM)
          #2 Bow tunnel thruster (RPM)             
          #3 Aft tunnel thruster (RPM)
          #4 Aft tunnel thruster (RPM)            
          #5 Right poontoon main propeller (RPM)
          $6 Left pontoon main propeller (RPM) ]

u_alloc = controlAllocation(tau)
    Control allocation based on the pseudoinverse                 

n = DPcontrol(eta,nu,sampleTime)
    Nonlinear PID controller for DP based on pole placement.    

n = stepInput(t) generates propellers step inputs n = [n1, n2, n3, n4, n5, n6].
    
Reference: 

    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
         Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
from python_vehicle_simulator.lib.control import DPpolePlacement
from python_vehicle_simulator.lib.gnc import sat, ssa

# Class Vehicle
class semisub:
    """
    semisub()                                      Propeller step inputs 
    semisub('DPcontrol',x_d,y_d,psi_d,V_c,beta_c)  DP control system
    
    Inputs:
        x_d: desired x position (m)
        y_d: desired y position (m)
        psi_d: desired yaw angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
    """

    # 可以考虑把这里的参数换成DP船中的对应参数 也就是MATLAB的RUN_DP函数中的

    def __init__(
        self,
        controlSystem="stepInput",
        r_x=0,
        r_y=0,
        r_n=0,
        V_current=0,
        beta_current=0,
    ):
        
        # Constants
        D2R = math.pi / 180                 # deg2rad

        if controlSystem == "DPcontrol":
            self.controlDescription = (
                "Nonlinear DP control (x_d, y_d, psi_d) = ("
                + str(r_x)
                + " m, "
                + str(r_y)
                + " m, "
                + str(r_n)
                + " deg)"
            )

        else:
            self.controlDescription = "Step inputs n = [n1, n2, n3, n4, n5, n6]"
            controlSystem = "stepInput"

        self.ref = np.array([r_x, r_y, r_n * D2R], float)
        self.V_c = V_current
        self.beta_c = beta_current * D2R
        self.controlMode = controlSystem

        # Initialize the semisub model
        self.L = 84.6   # Length (m)
        self.T_n = 1.0  # propeller revolution time constants (s)
        self.n_max = np.array(                      # RPM saturation limits (N)  
            [160, 160, 160, 160, 250, 250], float
        )                                     
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)        # velocity vector
        self.u_actual = np.array([0, 0, 0, 0, 0, 0], float)  # RPM inputs
        self.name = "Semisubmersible (see 'semisub.py' for more details)"

        self.controls = [
            "#1 Bow tunnel thruster (RPM)",
            "#2 Bow tunnel thruster (RPM)",
            "#3 Aft tunnel thruster (RPM)",
            "#4 Aft tunnel thruster (RPM)",
            "#5 Right poontoon main propeller (RPM)",
            "$6 Left pontoon main propeller (RPM)",
        ]
        self.dimU = len(self.controls)

        ## 原始版本
        # Semisub model
        MRB = 1.0e10 * np.array(
            [
                [0.0027, 0, 0, 0, -0.0530, 0],
                [0, 0.0027, 0, 0.0530, 0, -0.0014],
                [0, 0, 0.0027, 0, 0.0014, 0],
                [0, 0.0530, 0, 3.4775, 0, -0.0265],
                [-0.0530, 0, 0.0014, 0, 3.8150, 0],
                [0, -0.0014, 0, -0.0265, 0, 3.7192],
            ],
            float,
        )

        ### GMY change
        # MRB = 1.0e11 * np.array(
        #     [
        #         [0.001, 0, 0, 0, 0.001, 0],
        #         [0, 0.001, 0, -0.001, 0, 0],
        #         [0, 0, 0.001, 0, 0, 0],
        #         [0, -0.001, 0, 0.2668, 0, 0],
        #         [0.001, 0, 0, 0, 2.9280, 0],
        #         [0, 0, 0, 0, 0, 2.9280],
        #     ],
        #     float,
        # )

        ### 原始版本
        MA = 1.0e10 * np.array(
            [
                [0.0017, 0, 0, 0, -0.0255, 0],
                [0, 0.0042, 0, 0.0365, 0, 0],
                [0, 0, 0.0021, 0, 0, 0],
                [0, 0.0365, 0, 1.3416, 0, 0],
                [-0.0255, 0, 0, 0, 2.2267, 0],
                [0, 0, 0, 0, 0, 3.2049],
            ],
            float,
        )

        ### GMY change
        # MA = 1.0e11 * np.array(
        #     [
        #         [0, 0, -0, 0, -0.0033, 0],
        #         [0, 0.0002, 0, -0.0002, 0, 0.0003],
        #         [-0, 0, 0.0017, 0, -0.0004, 0],
        #         [0, -0.0002, 0, 0.0952, 0, -0.0021],
        #         [-0.0034, 0, -0.0003, 0, 3.9154, 0],
        #         [0, 0.0003, 0, -0.0023, 0, 0.5462],
        #     ],
        #     float,
        # )

        ## 原始版本
        self.D = 1.0e09 * np.array(
            [
                [0.0004, 0, 0, 0, -0.0085, 0],
                [0, 0.0003, 0, 0.0067, 0, -0.0002],
                [0, 0, 0.0034, 0, 0.0017, 0],
                [0, 0.0067, 0, 4.8841, 0, -0.0034],
                [-0.0085, 0, 0.0017, 0, 7.1383, 0],
                [0, -0.0002, 0, -0.0034, 0, 0.8656],
            ],
            float,
        )

        ### GMY change
        # self.D = 0 * np.array(
        #     [
        #         [0.0004, 0, 0, 0, -0.0085, 0],
        #         [0, 0.0003, 0, 0.0067, 0, -0.0002],
        #         [0, 0, 0.0034, 0, 0.0017, 0],
        #         [0, 0.0067, 0, 4.8841, 0, -0.0034],
        #         [-0.0085, 0, 0.0017, 0, 7.1383, 0],
        #         [0, -0.0002, 0, -0.0034, 0, 0.8656],
        #     ],
        #     float,
        # )

        ## 原始版本
        self.G = 1.0e10 * np.diag([0.0, 0.0, 0.0006, 1.4296, 2.6212, 0.0])

        ### GMY change
        # self.G = 1.0e11 * np.array(
        #     [
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 0.0008, 0, 0.0008, 0],
        #         [0, 0, 0, 0.0783, 0, 0],
        #         [0, 0, 0.0008, 0, 2.4766, 0],
        #         [0, 0, 0, 0, 0, 0],
        #     ],
        #     float,
        # )

        self.M = MRB + MA

        self.Minv = np.linalg.inv(self.M)

        # Thrust coefficient and configuration matrices (Fossen 2021, Ch. 11.2)
        K = np.diag([3.5, 3.5, 25.0, 25.0, 25.0, 25.0])
        T = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [30, 20, -20, -30, -self.L / 2, self.L / 2],
            ],
            float,
        )
        self.B = T @ K

        # DP control system
        self.e_int = np.array([0, 0, 0], float)  # integral states
        self.x_d = 0.0  # setpoints
        self.y_d = 0.0
        self.psi_d = 0.0
        self.wn = np.diag([0.15, 0.15, 0.05])  # PID pole placement
        self.zeta = np.diag([1.0, 1.0, 1.0])

        # GMY 存储期望轨迹变量: x_d, y_d, psi_d
        self.eta_d = np.array([0, 0, 0], float)
        self.eta_dot = np.array([0, 0, 0], float)
        self.eta_ddot = np.array([0, 0, 0], float)

        self.tau3 = np.array([0, 0, 0], float)
        self.y_hat = np.array([0, 0, 0], float)
        self.xi_hat = np.array([0, 0, 0, 0, 0, 0], float)
        self.b_hat = np.array([0, 0, 0], float)
        self.nu_hat = np.array([0, 0, 0], float)
        self.eta_hat = np.array([0, 0, 0], float)

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates the
        semisub equations of motion using Euler's method.
        """

        # Input vector
        n = u_actual

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge velocity
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway velocity

        # 调节流速（后续要求改到六自由度均有流速）
        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        nu_r = nu - nu_c  # relative velocity vector

        # Control forces and moments with propeller saturation
        n_squared = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = sat(n[i], -self.n_max[i], self.n_max[i])  # saturation limits
            n_squared[i] = abs(n[i]) * n[i]

        tau3 = np.matmul(self.B, n_squared)

        tau = np.array([tau3[0], tau3[1], 0, 0, 0, tau3[2]], float)

        # GMY change (直接用动力定位系统课程中代码求出的u)
        # tau = np.array([u_actual[0], u_actual[1], 0, 0, 0, u_actual[2]], float)

        # 6-DOF semisub model
        nu_dot = np.matmul(
            self.Minv, tau - np.matmul(self.D, nu_r) - np.matmul(self.G, eta)
        )
        n_dot = (u_control - u_actual) / self.T_n

        # Forward Euler integration
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot

        u_actual = np.array(n, float)

        return nu, u_actual


    def controlAllocation(self, tau3):
        """
        u_alloc  = controlAllocation(tau3),  tau3 = [tau_X, tau_Y, tau_N]'
        u_alloc = B' * inv( B * B' ) * tau3
        """

        ## GMY change
        # # 定义角度并转为弧度
        # alpha = np.array([90, 90, 90, 90, 0, 90, 0, 90]) * np.pi / 180
        #
        # # lx, ly 定义
        # lx = np.array([90, 70, -50, -70, 0, -90, 0, -90])
        # ly = np.array([0, 0, 0, 0, -15, 0, 15, 0])
        #
        # # 构建 T 矩阵
        # T = np.zeros((3, 8))
        # for i in range(8):
        #     T[0, i] = np.cos(alpha[i])
        #     T[1, i] = np.sin(alpha[i])
        #     T[2, i] = lx[i] * np.sin(alpha[i]) - ly[i] * np.cos(alpha[i])
        #
        # # Moore-Penrose 伪逆（等价于 T'*inv(T*T')）
        # T_dagger = T.T @ np.linalg.inv(T @ T.T)
        #
        # # 假设 tau 是 3x1 向量（你需要在实际使用中定义它）
        # # 例如：tau = np.array([tau1, tau2, tau3])
        # # tau = np.array([...])
        #
        # # 下面是计算 F 的部分
        # F = np.zeros(6)
        #
        # # FF 是 8x1 向量，结果为 T_dagger @ tau
        # FF = T_dagger @ tau3
        #
        # F[0:4] = FF[0:4]
        #
        # # 第五和第六元素根据向量模长计算
        # F[4] = np.sqrt(FF[4] ** 2 + FF[5] ** 2)
        # F[5] = np.sqrt(FF[6] ** 2 + FF[7] ** 2)
        #
        # alpha5 = np.degrees(np.arctan2(FF[5], FF[4]))
        # alpha6 = np.degrees(np.arctan2(FF[7], FF[6]))
        #
        # # 构建 alpha（单位为弧度）
        # alpha = np.radians([90, 90, 90, 90, alpha5, alpha6])
        #
        # # lx 和 ly 向量
        # lx = np.array([90, 70, -50, -70, -90, -90])
        # ly = np.array([0, 0, 0, 0, -15, 15])
        #
        # # 初始化 T 矩阵（3x6）
        # T = np.zeros((3, 6))
        #
        # # 构建 T 的每一列
        # for i in range(6):
        #     T[0, i] = np.cos(alpha[i])
        #     T[1, i] = np.sin(alpha[i])
        #     T[2, i] = lx[i] * np.sin(alpha[i]) - ly[i] * np.cos(alpha[i])
        # u_alloc = np.matmul(T, F)
        # u_alloc = np.array([u_alloc[0], u_alloc[1], u_alloc[2], 0, 0, 0])



        # # 原始版本
        B_pseudoInv = self.B.T @ np.linalg.inv(self.B @ self.B.T)
        u_alloc = np.matmul(B_pseudoInv, tau3)
        # print("A shape:", B_pseudoInv.shape)  # (6, 3)
        # print("B shape:", u_alloc)  # (6,)


        # GMY change （修改后变化很大）
        # coe = np.linalg.inv(K) @ T.T @ np.linalg.inv(T @ T.T)
        # u_alloc = coe @ tau3

        return u_alloc


    def DPcontrol(self, eta, nu, sampleTime):
        """
        u = DPcontrol(eta,nu,sampleTime) is a nonlinear PID controller
        for DP based on pole placement:

        tau = -R' Kp (eta-r) - R' Kd R nu - R' Ki int(eta-r)
        u = B_pseudoinverse * tau
        """
        # 3-DOF state vectors
        eta3 = np.array([eta[0], eta[1], eta[5]])
        nu3 = np.array([nu[0], nu[1], nu[5]])

        # 3-DOF diagonal model matrices
        M3 = np.diag([self.M[0][0], self.M[1][1], self.M[5][5]])
        D3 = np.diag([self.D[0][0], self.D[1][1], self.D[5][5]])

        # # GMY change2
        # [self.tau3, self.eta_d, self.eta_dot, self.eta_ddot, self.y_hat, self.xi_hat, self.b_hat, self.nu_hat, self.eta_hat] = DPpolePlacement(
        #     M3, D3, eta3,
        #     self.eta_d, self.eta_dot, self.eta_ddot,  # 参考系统状态
        #     self.wn, self.zeta, self.ref, sampleTime, self.tau3,
        #     self.y_hat, self.xi_hat, self.b_hat, self.nu_hat, self.eta_hat
        # )

        ##### GMY change
        [tau3, self.e_int, self.eta_d, self.eta_dot, self.eta_ddot] = DPpolePlacement(
            self.e_int,
            M3,
            D3,
            eta3,
            nu3,
            self.eta_d,
            self.eta_dot,
            self.eta_ddot,
            self.wn,
            self.zeta,
            self.ref,
            sampleTime,
        )


        #### 原始版本
        # [tau3, self.e_int, self.x_d, self.y_d, self.psi_d] = DPpolePlacement(
        #     self.e_int,
        #     M3,
        #     D3,
        #     eta3,
        #     nu3,
        #     self.x_d,
        #     self.y_d,
        #     self.psi_d,
        #     self.wn,
        #     self.zeta,
        #     self.ref,
        #     sampleTime,
        # )

        # u_alloc = self.controlAllocation(self.tau3)   # GMY change 原来是tau3

        #### 原始版本
        u_alloc = self.controlAllocation(tau3)


        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)

        ###### GMY change 把这部分内容注释了，替换为了 u_control = u_alloc
        ## 原始版本
        n = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = np.sign(u_alloc[i]) * math.sqrt(abs(u_alloc[i]))

        u_control = n

        # u_control = u_alloc

        return u_control

    def stepInput(self, t):
        """
        u = stepInput(t) generates propeller step inputs.
        """
        tau3 = np.array([10000, 0, 100000], float)

        if t > 30:
            tau3 = np.array([1000, 1000, 0], float)
        if t > 70:
            tau3 = np.array([0, 0, 0], float)

        u_control = self.controlAllocation(tau3)

        return u_control
