#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rl.py
------------------------------------
DQN-based Reinforcement Learning Trainer
for AI-DAC / Triple Loop Learning (TLL)
Loop 2: Policy Adaptation

Author: William K
Affiliation: University of Vienna
------------------------------------

This script implements:
 - DQN agent (vanilla Deep Q-Network)
 - Experience replay buffer
 - ε-greedy exploration
 - Target network updates
 - Training and evaluation loops

The environment is assumed to be Gym-like:
    obs = env.reset()
    obs, reward, done, info = env.step(action)

You can wrap your SQL/RDBMS anomaly detection policy
into such an environment (e.g., actions = responses
to anomaly scores, states = log features, context).
"""

import os
import math
import time
import random
import argparse
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))

        states = torch.stack(batch.state)
        actions = torch.tensor(batch.action, dtype=torch.long)
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        next_states = torch.stack(batch.next_state)
        dones = torch.tensor(batch.done, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 50_000,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> int:
        """
        ε-greedy policy.
        State must be 1D tensor on correct device.
        """
        if evaluate:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0))
                return int(q_values.argmax(dim=1).item())

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if random.random() < eps_threshold:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0))
                return int(q_values.argmax(dim=1).item())

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def maybe_update_target(self):
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------------------------------------------------------------------
# Training / Evaluation Loop
# ---------------------------------------------------------------------

def train_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int = 500,
    max_steps_per_episode: int = 500,
    warmup_steps: int = 1_000,
    log_interval: int = 10,
    model_save_path: str = "checkpoints/dqn_agent.pth",
):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    rewards_history = []
    losses_history = []

    global_step = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)

        episode_reward = 0.0
        episode_losses = []

        for t in range(max_steps_per_episode):
            global_step += 1

            # Select action
            action = agent.select_action(state, evaluate=False)

            # Step in environment
            next_state, reward, done, info = env.step(action)

            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=agent.device)
            reward_f = float(reward)
            done_f = float(done)

            # Store transition
            agent.replay_buffer.push(state, action, reward_f, next_state_t, done_f)
            state = next_state_t
            episode_reward += reward_f

            # Optimize model after warmup
            if global_step > warmup_steps:
                loss = agent.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)

                agent.maybe_update_target()

            if done:
                break

        rewards_history.append(episode_reward)
        if episode_losses:
            losses_history.append(np.mean(episode_losses))

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards_history[-log_interval:])
            avg_loss = np.mean(losses_history[-log_interval:]) if losses_history else float("nan")
            elapsed = time.time() - start_time
            print(
                f"[Episode {episode}/{num_episodes}] "
                f"AvgReward={avg_reward:.2f} AvgLoss={avg_loss:.4f} "
                f"Steps={global_step} Elapsed={elapsed:.1f}s"
            )

            # Save checkpoint
            torch.save(
                {
                    "policy_state_dict": agent.policy_net.state_dict(),
                    "target_state_dict": agent.target_net.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "steps_done": agent.steps_done,
                    "rewards_history": rewards_history,
                    "losses_history": losses_history,
                },
                model_save_path,
            )

    return rewards_history, losses_history


def evaluate_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int = 20,
    max_steps_per_episode: int = 500,
):
    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32, device=agent.device)
            episode_reward += float(reward)
            if done:
                break

        rewards.append(episode_reward)
        print(f"[Eval Episode {ep+1}/{num_episodes}] Reward={episode_reward:.2f}")

    print(f"[EVAL] MeanReward={np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    return rewards


# ---------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Trainer for RL-based Policy Adaptation (AI-DAC / TLL Loop 2)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--state-dim", type=int, default=32, help="State dimension (override if env not introspected)")
    parser.add_argument("--action-dim", type=int, default=4, help="Action dimension (override if env not introspected)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of DQN")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps-decay", type=int, default=50000, help="Epsilon decay steps")
    parser.add_argument("--target-update", type=int, default=1000, help="Target network update frequency (steps)")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Steps before learning begins")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (episodes)")
    parser.add_argument("--save-path", type=str, default="checkpoints/dqn_agent.pth", help="Path to save model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation with a saved model")
    return parser.parse_args()


def create_dummy_env(state_dim: int, action_dim: int):
    """
    Minimal synthetic env for testing wiring.
    Replace with your real RDBMS anomaly env.

    - State: random normal vector
    - Actions: {0..action_dim-1}
    - Reward: small random + bonus for specific action
    """

    class DummyEnv:
        def __init__(self, s_dim, a_dim):
            self.s_dim = s_dim
            self.a_dim = a_dim
            self.step_count = 0
            self.max_steps = 50

        def reset(self):
            self.step_count = 0
            return np.random.randn(self.s_dim).astype(np.float32)

        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.s_dim).astype(np.float32)
            reward = np.random.randn() * 0.01
            if action == 0:
                reward += 1.0  # arbitrary "good" action
            done = self.step_count >= self.max_steps
            info = {}
            return next_state, reward, done, info

    return DummyEnv(state_dim, action_dim)


def main():
    args = parse_args()

    # TODO: Replace create_dummy_env with your RDBMS / anomaly env factory
    env = create_dummy_env(args.state_dim, args.action_dim)

    # Infer dimensions if env exposes them
    state_example = env.reset()
    state_dim = state_example.shape[0]
    action_dim = args.action_dim

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        target_update_freq=args.target_update,
        device=args.device,
    )

    # Load for eval-only mode
    if args.eval_only and os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=args.device)
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.steps_done = ckpt.get("steps_done", 0)
        print(f"[INFO] Loaded checkpoint from {args.save_path}")

        evaluate_dqn(env, agent, num_episodes=20, max_steps_per_episode=args.max_steps)
        return

    # Train
    train_dqn(
        env,
        agent,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        model_save_path=args.save_path,
    )

    # Final evaluation
    print("[INFO] Training finished. Starting final evaluation...")
    evaluate_dqn(env, agent, num_episodes=20, max_steps_per_episode=args.max_steps)


if __name__ == "__main__":
    main()

