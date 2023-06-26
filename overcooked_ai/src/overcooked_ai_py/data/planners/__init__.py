import os
import pickle
import time
import numpy as np

from overcooked_ai_py.static import PLANNERS_DIR
from overcooked_ai_py.utils import load_dict_from_file


def load_saved_action_manager(filename):
    # sleep for a random number of seconds within range [0, 10] to handle concurrent file access
    # time.sleep(np.random.rand() * 5)
    with open(os.path.join(PLANNERS_DIR, filename), "rb") as f:
        mlp_action_manager = pickle.load(f)
        return mlp_action_manager


def load_saved_motion_planner(filename):
    # sleep for a random number of seconds within range [0, 10] to handle concurrent file access
    # time.sleep(np.random.rand() * 5)
    with open(os.path.join(PLANNERS_DIR, filename), "rb") as f:
        motion_planner = pickle.load(f)
        return motion_planner
