import gym
import numpy as np
import pybullet as p

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        # Correct observation space with velocity and orientation ranges
        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, 0, -5, -5, -5, -np.pi, -np.pi, -np.pi]),
            high=np.array([10, 10, 5, 5, 5, 5, np.pi, np.pi, np.pi]),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.drone = p.loadURDF("models/quadrotor.urdf", basePosition=[0, 0, 2])
        self.mass = p.getDynamicsInfo(self.drone, -1)[0]  # Get drone mass

        self.current_setpoint = 2  # Fixed setpoint for simplicity

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 2], [0, 0, 0, 1])
        p.resetBaseVelocity(self.drone, linearVelocity=[0, 0, -0.2])
        # Get initial state from simulation
        position, orientation = p.getBasePositionAndOrientation(self.drone)
        linear_velocity, _ = p.getBaseVelocity(self.drone)
        euler_angles = p.getEulerFromQuaternion(orientation)
        self.state = np.concatenate([position, linear_velocity, euler_angles])
        return self.state

    def step(self, action):
        thrust = action[0]
        # Apply thrust scaled by mass (to handle varying URDF masses)
        thrust_force = thrust * self.mass * 9.81  # Scale to counteract gravity
        p.applyExternalForce(
            self.drone, 
            linkIndex=-1, 
            forceObj=[0, 0, thrust_force],
            posObj=[0, 0, 0], 
            flags=p.WORLD_FRAME
        )
        p.stepSimulation()

        # Get new state
        new_position, new_orientation = p.getBasePositionAndOrientation(self.drone)
        linear_velocity, _ = p.getBaseVelocity(self.drone)
        euler_angles = p.getEulerFromQuaternion(new_orientation)
        next_state = np.concatenate([new_position, linear_velocity, euler_angles])

        # Calculate reward
        z = new_position[2]
        delta_z = abs(z - self.current_setpoint)
        vz = linear_velocity[2]
        reward = 10 - (delta_z ** 2) * 5 - abs(vz) * 2  # Removed thrust penalty

        # Done condition: out of bounds
        done = z <= 0 or z >= 4

        return next_state, reward, done, {}