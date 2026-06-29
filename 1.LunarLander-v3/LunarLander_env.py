# Hugging Face packages
import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env_id = "LunarLander-v3"
model_architecture = "PPO"

# We create our environment with gym.make("<name_of_the_environment>")
env = gym.make(env_id)
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())  # Get a random observation

print("\nObservation Space Vectors Description:\n"
    "Horizontal pad coordinate (x)\n"
    "Vertical pad coordinate (y)\n"
    "Horizontal speed (x)\n"
    "Vertical speed (y)\n"
    "Angle\n"
    "Angular speed\n"
    "If the left leg contact point has touched the land (boolean)\n"
    "If the right leg contact point has touched the land (boolean)\n")

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action

print("\nAction Space Vectors Description:\n"
    "Do nothing (0)\n"
    "Fire left engine (1)\n"
    "Fire main engine (2)\n"
    "Fire right engine (3)\n")

print("Reward system:\n"
    "For each step, the reward:\n"
    "Is increased/decreased the closer/further the lander is to the landing pad.\n"
    "Is increased/decreased the slower/faster the lander is moving.\n"
    "Is decreased the more the lander is tilted (angle not horizontal).\n"
    "Is increased by 10 points for each leg that is in contact with the ground.\n"
    "Is decreased by 0.03 points each frame a side engine is firing.\n"
    "Is decreased by 0.3 points each frame the main engine is firing.\n"
    "The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.\n"
    "An episode is considered a solution if it scores at least 200 points.\n")

# Create a vectorized environment to run multiple instances of the environment in parallel
# This is useful for training RL agents faster

number_envs = 16
print("Creating ", number_envs, " parallel environments for training...")
env = make_vec_env(env_id, n_envs=number_envs)  # Create n parallel environments


# Train the model using PPO algorithm
print("Creating the model using PPO algorithm...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024, # 1024
    batch_size=64, # 64
    n_epochs=16, # 4
    gamma=0.990, # 0.999
    gae_lambda=0.98, # 0.98
    ent_coef=0.05, # 0.01
    learning_rate=0.0005, # did not existed (0.0003)
    verbose=2, # 1
)

# Train the model for a certain number of timesteps
timesteps = 1000000
print("Training the model for ", timesteps, " timesteps...")
# Train it
model.learn(total_timesteps=timesteps)
# Save the model
model_name = "ppo-LunarLander-v3"
model.save(model_name)
print("Model saved as ", model_name)

# We now evaluate the model by running it in the environment and calculating the mean reward over a number of episodes
print("Evaluating the model...")
eval_env = Monitor(gym.make(env_id))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
