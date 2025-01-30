import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env import DroneEnv

# Store metrics
time_steps = []
heights = []
thrusts = []
rewards = []

def track_metrics(time, height, thrust, reward):
    time_steps.append(time)
    heights.append(height)
    thrusts.append(thrust)
    rewards.append(reward)

# Setup PyBullet
print("Setting up PyBullet...")
p.connect(p.DIRECT)  # Use p.GUI for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Create environment
print("Creating environment...")
env = DummyVecEnv([lambda: DroneEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Initialize PPO model
print("Initializing PPO model...")
model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,          # Adjusted for faster learning
    gamma=0.99,                  # Focus on long-term rewards
    clip_range=0.2,              # Allow larger updates
    ent_coef=0.001,              # Less exploration over time
    n_steps=2048,                # Smaller batch size
    policy_kwargs=dict(net_arch=[256, 256]),  # Slightly bigger network
    verbose=1,
    tensorboard_log="./ppo_drone_tensorboard/"
)

# Train the model
print("Training...")
model.learn(total_timesteps=100000)
model.save("ppo_drone_model")

# Save normalization stats
env.save("vec_normalize.pkl")

# Evaluation Phase
print("Evaluating...")
del env  # Remove training env

# Load normalization stats
eval_env = DummyVecEnv([lambda: DroneEnv()])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False  # Disable updates to normalization stats
eval_env.norm_reward = False  # Optional: disable reward normalization

# Load the trained model
model = PPO.load("ppo_drone_model", env=eval_env)

# Run evaluation
obs = eval_env.reset()
for step in range(1000):  # Short evaluation loop
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = eval_env.step(action)
    height = obs[0][2]  # Extract height from observation
    thrust = action[0]  # Extract thrust from action
    track_metrics(step, height, thrust, reward)
    if done:
        obs = eval_env.reset()

# Plot metrics
plt.figure(figsize=(12, 8))

# Height vs Time
plt.subplot(3, 1, 1)
plt.plot(time_steps, heights, label="Height")
plt.xlabel('Time Step')
plt.ylabel('Height')
plt.title('Drone Height vs Time')

# Thrust vs Time
plt.subplot(3, 1, 2)
plt.plot(time_steps, thrusts, label="Thrust", color='r')
plt.xlabel('Time Step')
plt.ylabel('Thrust')
plt.title('Thrust vs Time')

# Reward vs Time
plt.subplot(3, 1, 3)
plt.plot(time_steps, rewards, label="Reward", color='g')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Reward vs Time')

plt.tight_layout()
plt.show()