import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# Define a simple environment for testing
class SimpleEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 1:
            self.state = 1
        else:
            self.state = 0
        return self.state, 1.0, False, {}


# Create the environment
env = DummyVecEnv([lambda: SimpleEnvironment()])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_simple_model")

# Evaluate the trained model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
