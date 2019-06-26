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

# features=np.ones(10,)
# cycle_durations = np.arange(.1,10,.1)
# test1_no = cycle_durations.shape[0]
# exp1_average_error = np.zeros([2,test1_no]) # first row open-loop and second row close-loop

# #cycle length experiment
# for cycle_duration_in_seconds, ii in zip(cycle_durations, range(test1_no)):
# 	[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = cycle_duration_in_seconds, show=False)
# 	desired_kinematics = \
# 	positions_to_kinematics_fcn(q0_filtered, q1_filtered, timestep = 0.005)
# 	exp1_average_error[0,ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
# 	exp1_average_error[1,ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False) # K = [10, 15]

# test2_no = 50
# exp2_average_error = np.zeros([2,test2_no])
# for ii in range(test2_no):
# 	features = np.random.rand(10)*.8+.2
# 	[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 1.3, show=False)
# 	desired_kinematics = positions_to_kinematics_fcn(q0_filtered, q1_filtered, timestep = 0.005)
# 	exp2_average_error[0,ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
# 	exp2_average_error[1,ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False) # K = [10, 15]


# test3_no = 50
# exp3_average_error = np.zeros([2,test3_no])
# for ii in range(test3_no):
# 	q0 = p2p_positions_gen_fcn(low=-np.pi/3, high=np.pi/3, number_of_positions=5, duration_of_each_position=3, timestep=.005)
# 	q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=0, number_of_positions=5, duration_of_each_position=3, timestep=.005)
# 	desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
# 	exp3_average_error[0,ii] = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
# 	exp3_average_error[1,ii] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], plot_outputs=False, Mj_render=False) # K = [10, 15]

# # plotting the results
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
# exp4_average_error[1,:] = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[10, 15], model_ver=2, plot_outputs=True, Mj_render=True)
# import pdb; pdb.set_trace()
# # errors_K = [exp1_average_error, exp2_average_error, exp3_average_error, exp4_average_error]
# pickle.dump([errors_K],open("results/feedback_K.sav", 'wb')) # saving the results with only K
[errors_K] = pickle.load(open("results/feedback_K.sav", 'rb')) # loading the results with only K
plt.figure()
plt.plot(np.arange(.1,10,.1), errors_K[0][0,:], np.arange(.1,10,.1), errors_K[0][1,:])
plt.show(block=True)
plt.figure()
plt.plot(range(errors_K[1][0,:].shape[0]), errors_K[1][0,:], range(errors_K[1][0,:].shape[0]), errors_K[1][1,:])
plt.show(block=True)
plt.figure()
plt.plot(range(errors_K[2][0,:].shape[0]), errors_K[2][0,:], range(errors_K[2][0,:].shape[0]), errors_K[2][1,:])
plt.show(block=True)

#import pdb; pdb.set_trace()
