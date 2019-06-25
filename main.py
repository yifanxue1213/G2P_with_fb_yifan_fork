
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

features=np.ones(10,)
cycle_durations = np.arange(.1,10,.1)
test1_no = cycle_durations.shape[0]
exp1_average_error_o = np.zeros(test1_no,)
exp1_average_error_c = np.zeros(test1_no,)
# cycle length experiment
for cycle_duration_in_seconds, ii in zip(cycle_durations, range(test1_no)):
	[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = cycle_duration_in_seconds, show=False)
	desired_kinematics = \
	positions_to_kinematics_fcn(q0_filtered, q1_filtered, timestep = 0.005)
	exp1_average_error_o[ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
	exp1_average_error_c[ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False) # K = [10, 15]
plt.figure()
plt.plot(cycle_durations, exp1_average_error_o, cycle_durations, exp1_average_error_c)
plt.show(block=True)

test2_no = 50
exp2_average_error_o = np.zeros(test2_no,)
exp2_average_error_c = np.zeros(test2_no,)
for ii in range(test2_no):
	features = np.random.rand(10)*.8+.2
	[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 1.3, show=False)
	desired_kinematics = positions_to_kinematics_fcn(q0_filtered, q1_filtered, timestep = 0.005)
	exp2_average_error_o[ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
	exp2_average_error_c[ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False) # K = [10, 15]
plt.figure()
plt.plot(range(test2_no), exp2_average_error_o, range(test2_no), exp2_average_error_c)
plt.show(block=True)

test3_no = 50
exp3_average_error_o = np.zeros(test3_no,)
exp3_average_error_c = np.zeros(test3_no,)
for ii in range(test3_no):
	q0 = p2p_positions_gen_fcn(low=-np.pi/3, high=np.pi/3, number_of_positions=5, duration_of_each_position=3, timestep=.005)
	q1 = p2p_positions_gen_fcn(low=-np.pi, high=0, number_of_positions=5, duration_of_each_position=3, timestep=.005)
	desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
	exp3_average_error_o[ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
	exp3_average_error_c[ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False) # K = [10, 15]
plt.figure()
plt.plot(range(test3_no), exp3_average_error_o, range(test3_no), exp3_average_error_c)
plt.show(block=True)
#import pdb; pdb.set_trace()
