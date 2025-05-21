from stable_baselines3 import DQN

# Load the trained model
model = DQN.load("rl/policy_dqn.zip")

# Example: simulate a decision
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
print("Selected action:", action)
