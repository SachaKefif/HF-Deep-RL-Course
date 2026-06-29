# Publish to Hugging Face Leaderboard
# Command : huggingface-cli login

from types import SimpleNamespace

import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from huggingface_sb3 import package_to_hub

from LunarLander_env import env_id, model, model_architecture, model_name

# Compatibility fix:
# huggingface_sb3 expects older SB3 recorders to expose env.video_recorder.path.
# Current SB3 exposes env.video_path instead.
if not hasattr(VecVideoRecorder, "video_recorder"):
    VecVideoRecorder.video_recorder = property(
        lambda self: SimpleNamespace(path=self.video_path)
    )

## Define a repo_id
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
## CHANGE WITH YOUR REPO ID
repo_id = "sachaii/ppo-LunarLander-v3"  # Change with your repo id, you can't push with mine 😄

## Define the commit message
commit_message = "Upload PPO LunarLander-v3 trained agent"

# Create the evaluation env and set the render_mode="rgb_array"
eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

# PLACE the package_to_hub function you've just filled here
package_to_hub(
    model=model,  # Our trained model
    model_name=model_name,  # The name of our trained model
    model_architecture=model_architecture,  # The model architecture we used: in our case PPO
    env_id=env_id,  # Name of the environment
    eval_env=eval_env,  # Evaluation Environment
    repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
    commit_message=commit_message,
)
