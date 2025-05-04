import numpy as np
import torch

def mpc_control(
    model,
    frames_buffer,
    actions_buffer,
    device,
    horizon=10,
    candidates=100,
    gamma=0.95,
    angle_soft_threshold=0.15,
    angle_soft_penalty=2.0
):
    best_cost = float('inf')
    best_action = None

    for _ in range(candidates):
        # Clone input buffers
        curr_frames = [f.copy() for f in frames_buffer]
        curr_actions = [a.copy() for a in actions_buffer]
        cumulative_cost = 0.0

        for t in range(horizon):
            # Sample a random action: 0 or 1
            action = [np.random.choice([0, 1])]
            curr_actions.pop(0)
            curr_actions.append(action)

            images_input = torch.tensor([curr_frames], dtype=torch.float32).to(device)
            actions_input = torch.tensor([curr_actions], dtype=torch.float32).to(device)

            with torch.no_grad():
                pred_state = model(images_input, actions_input).cpu().numpy()[0]

            # Compute cost components
            cart_pos = pred_state[0]
            cart_vel = pred_state[1]
            pole_angle = pred_state[2]
            pole_ang_vel = pred_state[3]

            cost = (
                1.0 * cart_pos**2 +
                0.1 * cart_vel**2 +
                20.0 * pole_angle**2 +        # Strong weight on angle
                0.1 * pole_ang_vel**2
            )

            # Add soft constraint penalty if near terminal angle threshold
            if abs(pole_angle) > angle_soft_threshold:
                cost += angle_soft_penalty

            cumulative_cost += (gamma ** t) * cost

            # Simulate new frame (assume it doesn't change — placeholder)
            # You can optionally simulate predicted next frame here
            # For now we keep the same frames_buffer as placeholder
            # If your model supports it, replace curr_frames[-1] with a decoded frame

        # Save the first action of the best trajectory
        if cumulative_cost < best_cost:
            best_cost = cumulative_cost
            best_action = curr_actions[-1][0]  # Get the last (most recent) action

    return best_action



def mpc_control_batched(
    model,
    frames_buffer,
    actions_buffer,
    device,
    candidates=100,
    horizon=10,
    gamma=0.95,
    angle_soft_threshold=0.05,
    angle_soft_penalty=25,
    pole_termination_limit=0.209
):
    action_dim = 1
    # Generate all candidate sequences: [candidates, horizon, 1]
    action_sequences = torch.randint(
        low=0, high=2, size=(candidates, horizon, action_dim), dtype=torch.float32
    ).to(device)

    # Repeat input sequence for all candidates
    frames_np = np.array(frames_buffer, dtype=np.float32)  # shape: [seq, C, H, W]
    frames_np = np.expand_dims(frames_np, axis=0)          # shape: [1, seq, C, H, W]
    frames_np = np.repeat(frames_np, candidates, axis=0)   # shape: [candidates, seq, C, H, W]
    frames_batch = torch.from_numpy(frames_np).to(device)





    actions_batch = torch.tensor([actions_buffer] * candidates, dtype=torch.float32).to(device)

    cumulative_costs = torch.zeros(candidates, dtype=torch.float32, device=device)

    for t in range(horizon):
        # Append new action from candidate sequence
        new_action = action_sequences[:, t:t+1, :]  # shape: [candidates, 1, 1]
        actions_batch = torch.cat([actions_batch[:, 1:, :], new_action], dim=1)

        with torch.no_grad():
            pred_states = model(frames_batch, actions_batch)  # [candidates, 4]

        # Extract state components
        cart_pos = pred_states[:, 0]
        cart_vel = pred_states[:, 1]
        pole_angle = pred_states[:, 2]
        ang_vel = pred_states[:, 3]

        # Base cost
        cost = (
            1.0 * cart_pos**2 +
            0.1 * cart_vel**2 +
            40.0 * pole_angle**2 +
            0.1 * ang_vel**2 # encourages oscillatory correction
        )


        # Soft constraint: penalize angle nearing termination
        over_soft_thresh = pole_angle.abs() > angle_soft_threshold
        soft_penalty = torch.zeros_like(cost)
        soft_penalty[over_soft_thresh] = angle_soft_penalty * (
            (pole_angle.abs()[over_soft_thresh] - angle_soft_threshold) /
            (pole_termination_limit - angle_soft_threshold)
        )
        cost += soft_penalty

        # Optionally: HARD penalty for violating real termination limit
        # early_termination = pole_angle.abs() > pole_termination_limit
        # cost[early_termination] += 1e6

        cumulative_costs += (gamma ** t) * cost

    # Pick best candidate
    best_idx = torch.argmin(cumulative_costs)
    best_action = action_sequences[best_idx, 0, 0].item()
    return int(round(best_action))


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(0, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self, measurement, dt=0.02):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        control = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )
        control = max(self.output_limits[0], min(control, self.output_limits[1]))
        self.prev_error = error
        return control


# ---------------------------
# PID-Based Control Wrapper
# ---------------------------
def pid_control(obs, controller: PIDController, dt=0.02):
    """
    PID-based control using pole angle.

    Args:
        obs (np.ndarray): current state (length 4)
        controller (PIDController): PID instance
        dt (float): time step

    Returns:
        action (int): discrete action (0 or 1)
    """
    pole_angle = obs[2]
    action_float = controller(pole_angle, dt=dt)
    return int(round(action_float))

def cem_control(
    model,
    frames_buffer,
    actions_buffer,
    device,
    horizon=10,
    candidates=100,
    iterations=4,
    elite_frac=0.2,
    gamma=0.95,
    angle_soft_threshold=0.1,
    angle_soft_penalty=40,
    pole_termination_limit=0.209
):
    action_dim = 1
    elite_count = int(candidates * elite_frac)
    probs = torch.full((horizon, action_dim), 0.5, device=device)  # initialize uniform

    for i in range(iterations):
        # Sample action sequences from Bernoulli
        action_sequences = torch.bernoulli(probs.expand(candidates, -1, -1)).to(device)

        # Prepare input buffers
        frames_np = np.array(frames_buffer, dtype=np.float32)
        frames_np = np.expand_dims(frames_np, axis=0)
        frames_np = np.repeat(frames_np, candidates, axis=0)
        frames_batch = torch.from_numpy(frames_np).to(device)

        actions_batch = torch.tensor([actions_buffer] * candidates, dtype=torch.float32).to(device)
        cumulative_costs = torch.zeros(candidates, dtype=torch.float32, device=device)

        for t in range(horizon):
            new_action = action_sequences[:, t:t+1, :]
            actions_batch = torch.cat([actions_batch[:, 1:, :], new_action], dim=1)

            with torch.no_grad():
                pred_states = model(frames_batch, actions_batch)

            cart_pos = pred_states[:, 0]
            cart_vel = pred_states[:, 1]
            pole_angle = pred_states[:, 2]
            ang_vel = pred_states[:, 3]

            cost = (
                5.0 * cart_pos**2 +
                .01 * cart_vel**2 +
                50.0 * pole_angle**2 +
                0.001 * ang_vel**2
            )

            # Soft constraint
            over_soft_thresh = pole_angle.abs() > angle_soft_threshold
            soft_penalty = torch.zeros_like(cost)
            soft_penalty[over_soft_thresh] = angle_soft_penalty * (
                (pole_angle.abs()[over_soft_thresh] - angle_soft_threshold) /
                (pole_termination_limit - angle_soft_threshold)
            )
            cost += soft_penalty

            cumulative_costs += (gamma ** t) * cost

        # Select elites and update probs
        elite_idxs = torch.topk(cumulative_costs, elite_count, largest=False).indices
        elite_actions = action_sequences[elite_idxs]
        probs = elite_actions.float().mean(dim=0)

    # Return first action of best elite sequence
    best_seq = elite_actions[0]
    return int(round(best_seq[0, 0].item()))
