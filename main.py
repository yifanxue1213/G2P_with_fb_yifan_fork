
import numpy as np
from matplotlib import pyplot as plt
import pickle
from warnings import simplefilter
from all_functions import *
from feedback_functions import *

simplefilter(action='ignore', category=FutureWarning)

# [babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
# model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)
# cum_kinematics = babbling_kinematics
# cum_activations = babbling_activations


# np.random.seed(0)
# pickle.dump([model,cum_kinematics, cum_activations],open("results/mlp_model.sav", 'wb'))

[model,cum_kinematics, cum_activations] = pickle.load(open("results/mlp_model.sav", 'rb')) # loading the model

desired_kinematics = create_sin_cos_kinematics_fcn(attempt_length = 10 , number_of_cycles = 7, timestep = 0.005)
average_error = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
print("average open-loop error is: ", average_error)
average_error = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False)
print("average close-loop error is: ", average_error)
#import pdb; pdb.set_trace()



