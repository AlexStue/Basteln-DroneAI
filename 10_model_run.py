import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_env import DroneEnv

# python3 10-model-train.py

print("------------------------------------------")
print("Setup PyBullet:")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

print("------------------------------------------")
print("Load and Sim:")

env = DummyVecEnv([lambda: DroneEnv()])
model = PPO.load("ppo_drone_model", env=env)

state = env.reset()

for _ in range(20000):
    action, _states = model.predict(state)
    state, reward, done, info = env.step(action)

    p.stepSimulation()

    if done:
        state = env.reset()

print("------------------------------------------")
print("Simulation completed.")
