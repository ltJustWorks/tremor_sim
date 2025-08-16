import numpy as np
import matplotlib.pyplot as plt

def compute_gate_value(X_PD):
    if not (0 <= X_PD <= 1):
        raise ValueError("X_PD must be between 0 and 1")
    return 1 - X_PD

class KalmanFilter:
    def __init__(self, H, D, Omega_measurement, Omega_xi, Omega_eta, x_est_init, P_init):
        self.H = H
        self.D = D
        self.Omega_measurement = Omega_measurement
        self.Omega_xi = Omega_xi  # process noise covariance
        self.Omega_eta = Omega_eta  # estimator noise covariance
        self.x_est = x_est_init.copy()
        self.P = P_init.copy()
    
    def step(self, A_k, B_k, u, y):
        innov_cov = self.H @ self.P @ self.H.T + self.Omega_measurement
        K = A_k @ self.P @ self.H.T @ np.linalg.inv(innov_cov)
        innovation = y - (self.H @ self.x_est + self.D * u)
        eta = np.random.multivariate_normal(np.zeros(self.x_est.shape[0]), self.Omega_eta)
        self.x_est = A_k @ self.x_est + B_k * u + K @ innovation + eta
        self.P = self.Omega_xi + self.Omega_eta + (A_k - K @ self.H) @ self.P @ A_k.T
        return self.x_est
    
class SDOFJointModel:
    def __init__(self, J, G, K, tau1, tau2, b, dt, noise_cov=None):
        self.J = J
        self.G = G
        self.K = K
        self.tau1 = tau1
        self.tau2 = tau2
        self.b = b
        self.dt = dt
        self.n = 7

        # state dynamics vector
        self.A = np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [-K/J, -G/J, 1/J, 0, 1/J, 0, 0],
            [0, 0, -1/tau2, 1/tau2, 0, 0, 0],
            [0, 0, 0, -1/tau1, 0, self.b/tau1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        # control input vector
        self.B = np.array([[0], [0], [0], [self.b/tau1], [0], [0], [0]])

        self.Ak = np.eye(self.n) + self.A * dt
        self.Bk = dt * (self.Ak @ self.B)

        if noise_cov is None:
            self.noise_cov = np.zeros((self.n, self.n))
        else:
            self.noise_cov = noise_cov

    def step(self, x, u):
        x_next = self.Ak @ x + self.Bk.flatten() * u
        noise = np.random.multivariate_normal(np.zeros(self.n), self.noise_cov)
        return x_next + noise

class ForwardModelKalmanFilter:
    def __init__(self, b, Omega_forward, Omega_xi, Omega_eta, x_est_init, P_init):
        self.b = b
        self.Omega_forward = Omega_forward
        self.Omega_xi = Omega_xi
        self.Omega_eta = Omega_eta
        self.x_est = x_est_init.copy()
        self.P = P_init.copy()
        self.H_f = np.array([[0, 0, 0, 0, 0, b, 0]])
        self.D_f = b

    def step(self, A_k, B_k, u, y):
        PHt = self.P @ self.H_f.T
        S = self.H_f @ self.P @ self.H_f.T + self.Omega_forward
        K = A_k @ PHt @ np.linalg.inv(S)
        innovation = y - (self.H_f @ self.x_est + self.D_f * u)
        eta = np.random.multivariate_normal(np.zeros(self.x_est.shape[0]), self.Omega_eta)
        self.x_est = A_k @ self.x_est + B_k * u + K.flatten() * innovation + eta
        self.P = self.Omega_xi + self.Omega_eta + (A_k - K @ self.H_f) @ self.P @ A_k.T
        return self.x_est

    def run_forward_model(self, A_k, B_k, control_inputs, measurements):
        for i in range(len(control_inputs)):
            u = control_inputs[i]
            y = measurements[i]
            self.step(A_k, B_k, u, y)
        return self.x_est

def compute_optimal_feedback_gain(A, B, Q, R, horizon):
    S = Q.copy()
    for _ in range(horizon):
        BT_S_B = B.T @ S @ B
        inv_term = np.linalg.inv(R + BT_S_B)
        L = inv_term @ (B.T @ S @ A)
        S = Q + A.T @ S @ (A - B @ L)
    return L

class SFFCController:
    def __init__(self, Q, R, horizon, open_loop_trajectory, b=1.0):
        self.Q = Q
        if np.isscalar(R):
            self.R = np.array([[R]])
        else:
            self.R = R
        self.horizon = horizon
        self.open_loop_trajectory = open_loop_trajectory
        self.b = b

    def compute_feedback_gain(self, A, B):
        return compute_optimal_feedback_gain(A, B, self.Q, self.R, self.horizon)

    def control_input(self, x_est, t, A, B):
        if t < len(self.open_loop_trajectory):
            u_OL = self.open_loop_trajectory[t]
        else:
            u_OL = 0.0
        L = self.compute_feedback_gain(A, B)
        u_FB = - (L @ x_est)[0]
        u_total = self.b * (u_OL + u_FB)
        return u_total, u_OL, u_FB


def simulate_simulation(sim_type, X_PD, total_steps=1000):
    dt = 0.001 # 1 ms time step
    delay_steps = 100 # 100 ms delay
    
    # system parameters
    J = 0.00276
    G = 0.03
    K = 0.992
    tau1 = 0.04
    tau2 = 0.04

    b_actual = compute_gate_value(X_PD)
    # but the internal model assumes b_internal = 1
    b_internal = 1.0

    # process noise covariance
    Omega_xi = np.diag([1.75e-7, 8.53e-7, 1.16e-7, 1.39e-7, 1.16e-6, 1.39e-6, 0])
    # measurement noise covariance
    Omega_omega = np.diag([1.75e-4, 8.53e-4, 1.16e-4, 1.39e-4])
    
    model = SDOFJointModel(J, G, K, tau1, tau2, b_actual, dt, noise_cov=Omega_xi)

    if sim_type == 'reaching':
        target = 1.0
        x0 = np.array([0.0, 0, 0, 0, 0, 0, target])
        uOL_traj = np.linspace(0.5, 0, total_steps)
        R_controller = 5e-2
    else:
        target = 0.0
        x0 = np.array([0.05, 0, 0, 0, 0, 0, target])
        uOL_traj = np.zeros(total_steps)
        R_controller = 1e-4

    H_est = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, -b_internal, 0]])
    D_est = np.array([0, 0, 0, b_internal])
    kf = KalmanFilter(H_est, D_est, Omega_omega, Omega_xi, Omega_xi, x0, np.eye(7)*1e-3)
    
    q_vec = np.array([1, 0, 0, 0, 0, 0, -1])
    Q = np.outer(q_vec, q_vec)
    horizon = 500
    controller = SFFCController(Q, R_controller, horizon, uOL_traj, b=b_internal)

    Omega_forward = np.array([[1e-4]])

    time_array = np.arange(total_steps+1) * dt
    theta_true = np.zeros(total_steps+1)
    theta_est_array = np.zeros(total_steps+1)
    u_array = np.zeros(total_steps)

    x_true = x0.copy()
    theta_true[0] = x_true[0]
    theta_est_array[0] = x0[0]

    est_history = [x0.copy()]
    control_history = []

    A_k = model.Ak
    B_k = model.Bk.flatten()
    
    # main loop
    for t in range(total_steps):
        # compensate for measurement delay
        if t < delay_steps:
            state_for_control = kf.x_est.copy()
        else:
            # obtain delayed estimator state
            delayed_state = est_history[t - delay_steps].copy()
            # forward model filter starting from the delayed state
            fm_filter = ForwardModelKalmanFilter(b_internal, Omega_forward, Omega_xi, Omega_xi, delayed_state, np.eye(7)*1e-3)
            # get stored control inputs over the delay period
            fm_control_inputs = control_history[t - delay_steps : t]
            fm_measurements = []
            # for each past control input, simulate an efferent copy measurement
            for u_val in fm_control_inputs:
                noise = np.random.normal(0, np.sqrt(Omega_forward[0,0]))
                y_val = b_internal * u_val + noise
                fm_measurements.append(y_val)
            # run forward model over these delay steps
            state_forward = fm_filter.run_forward_model(A_k, B_k, fm_control_inputs, fm_measurements)
            state_for_control = state_forward.copy()
        
        # compute overall control input
        u, u_OL, u_FB = controller.control_input(state_for_control, t, A_k, B_k.reshape(-1,1))
        # add saturation
        u = np.clip(u, -2, 2)
        u_array[t] = u
        control_history.append(u)
        
        # update the actual state
        x_true = model.step(x_true, u)
        theta_true[t+1] = x_true[0]
        
        # generate a measurement using the actual (system) parameters
        H_true = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -b_actual, 0]
        ])
        D_true = np.array([0, 0, 0, b_actual])
        noise_meas = np.random.multivariate_normal(np.zeros(4), Omega_omega)
        y = H_true @ x_true + D_true * u + noise_meas
        
        # update estimator
        x_est_new = kf.step(A_k, B_k, u, y)
        est_history.append(x_est_new.copy())
        theta_est_array[t+1] = x_est_new[0]
    
    return time_array, theta_true, theta_est_array, u_array


pd_levels = [0.0, 0.25, 0.5, 0.75, 1.0]  # healthy to maximum severity

plt.figure(figsize=(12, 6))
for X_PD in pd_levels:
    time_arr, theta_true, theta_est, u_arr = simulate_simulation('reaching', X_PD)
    plt.plot(time_arr, theta_true, label=f'X_PD = {X_PD:.2f}')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.title('Reaching Simulation: True Joint Angle for Varying PD Levels')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for X_PD in pd_levels:
    time_arr, theta_true, theta_est, u_arr = simulate_simulation('rest', X_PD)
    plt.plot(time_arr, theta_true, label=f'X_PD = {X_PD:.2f}')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.title('Rest Simulation: True Joint Angle for Varying PD Levels')
plt.legend()
plt.grid(True)
plt.show()
