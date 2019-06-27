import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from warnings import simplefilter
from all_functions import *
from feedback_functions import *

simplefilter(action='ignore', category=FutureWarning)

# # [babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
# # model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)
# # cum_kinematics = babbling_kinematics
# # cum_activations = babbling_activations



# # pickle.dump([model,cum_kinematics, cum_activations],open("results/mlp_model.sav", 'wb'))

# [model,cum_kinematics, cum_activations] = pickle.load(open("results/mlp_model.sav", 'rb')) # loading the model
# np.random.seed(0)

# P = [10, 15]
# I = [0, 0]

# trial_number = 50

# features=np.ones(10,)
# cycle_durations = np.linspace(.1,10,trial_number)
# test1_no = cycle_durations.shape[0]
# exp1_average_error = np.zeros([2,test1_no]) # first row open-loop and second row close-loop
# #cycle length experiment
# for cycle_duration_in_seconds, ii in zip(cycle_durations, range(test1_no)):
# 	[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = cycle_duration_in_seconds, show=False)
# 	q0_filtered_10 = np.tile(q0_filtered,10)
# 	q1_filtered_10 = np.tile(q1_filtered,10)
# 	desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)
# 	exp1_average_error[0,ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
# 	exp1_average_error[1,ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]

# test2_no = trial_number
# exp2_average_error = np.zeros([2,test2_no])
# for ii in range(test2_no):
# 	features = np.random.rand(10)*.8+.2
# 	[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 3.3, show=False)
# 	#import pdb; pdb.set_trace()
# 	q0_filtered_10 = np.tile(q0_filtered,10)
# 	q1_filtered_10 = np.tile(q1_filtered,10)
# 	desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)
# 	exp2_average_error[0,ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
# 	exp2_average_error[1,ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]
# 	#print("error_without: ", exp2_average_error[0,0], "error with: ", exp2_average_error[1,0])

# test3_no = trial_number
# exp3_average_error = np.zeros([2,test3_no])
# for ii in range(test3_no):
# 	q0 = p2p_positions_gen_fcn(low=-np.pi/3, high=np.pi/3, number_of_positions=10, duration_of_each_position=3, timestep=.005)
# 	q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=0, number_of_positions=10, duration_of_each_position=3, timestep=.005)
# 	desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
# 	exp3_average_error[0,ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
# 	exp3_average_error[1,ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics,  P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]


# test4_no = 1
# exp4_average_error = np.zeros([2,test4_no])
# q0 = p2p_positions_gen_fcn(low=np.pi/3, high=np.pi/3, number_of_positions=1, duration_of_each_position=1, timestep=.005)
# q0 = np.append(q0,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=14, timestep=.005))
# q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=-np.pi/2, number_of_positions=1, duration_of_each_position=1, timestep=.005)
# q1 = np.append(q1,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=14, timestep=.005))
# desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
# exp4_average_error[0,:] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, model_ver=2, plot_outputs=True, Mj_render=True)
# q0 = p2p_positions_gen_fcn(low=np.pi/3, high=np.pi/3, number_of_positions=1, duration_of_each_position=1, timestep=.005)
# q0 = np.append(q0,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=4, timestep=.005))
# q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=-np.pi/2, number_of_positions=1, duration_of_each_position=1, timestep=.005)
# q1 = np.append(q1,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=4, timestep=.005))
# desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
# exp4_average_error[1,:] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics,  P=P, I=I, model_ver=2, plot_outputs=True, Mj_render=True)

# errors_all = [exp1_average_error, exp2_average_error, exp3_average_error, exp4_average_error]
# pickle.dump([errors_all],open("results/feedback_errors_.sav", 'wb')) # saving the results with only P
[errors_all] = pickle.load(open("results/feedback_errors_P_I.sav", 'rb')) # loading the results with only P
plot_comparison_figures_fcn(errors_all)

#import pdb; pdb.set_trace()
