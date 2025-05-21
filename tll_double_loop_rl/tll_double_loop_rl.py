
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from gym import Env
from gym.spaces import Discrete, Box

class SimpleSQLDetectionEnv(Env):
    def __init__(self, df):
        super(SimpleSQLDetectionEnv, self).__init__()
        self.df = df
        self.index = 0
        self.action_space = Discrete(2)  # 0: benign, 1: anomaly
        self.observation_space = Box(low=0, high=1, shape=(df.shape[1]-1,), dtype=np.float32)

    def reset(self):
        self.index = 0
        return self.df.iloc[self.index, :-1].values.astype(np.float32)

    def step(self, action):
        label = self.df.iloc[self.index, -1]
        reward = 1 if action == label else -1
        self.index += 1
        done = self.index >= len(self.df)
        obs = self.df.iloc[self.index, :-1].values.astype(np.float32) if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

# Load dataset
df = pd.read_csv("sql_train.csv")
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
X = df.drop("label", axis=1)
y = df["label"]
df_combined = pd.concat([X, y], axis=1)

# Initialize environment
env = SimpleSQLDetectionEnv(df_combined)
check_env(env)

# Train agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tll_drl_tensorboard/")
model.learn(total_timesteps=100000)

# Save model
model.save("drl_sql_detector")
