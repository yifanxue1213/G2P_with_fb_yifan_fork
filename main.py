
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

[babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)
cum_kinematics = babbling_kinematics
cum_activations = babbling_activations

[model, errors, cum_kinematics, cum_activations] =\
	in_air_adaptation_fcn(
		model=model,
		babbling_kinematics=cum_kinematics,
		babbling_activations=cum_activations,
		number_of_refinements=3,
		Mj_render=True)