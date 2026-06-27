# Dependencies
# NOTE: install dependencies from your shell, not inside this Python script.
# Example:
#   pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt
#   pip install pyvirtualdisplay stable-baselines3 gymnasium huggingface_sb3 huggingface_hub

# Virtual display
# pyvirtualdisplay needs Xvfb, which is not installed on native Windows.
import os

if os.name != "nt":
    from pyvirtualdisplay import Display

    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

# Hugging Face packages
import gymnasium

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor











































