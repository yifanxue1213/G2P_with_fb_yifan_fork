
import numpy as np
from matplotlib import pyplot as plt
import pickle
from warnings import simplefilter
from all_functions import *
from feedback_functions import *

def calculate_inputkinematics(step_number, real_attempt_positions, desired_kinematics, K, timestep=.005):
	#import pdb; pdb.set_trace()


	#import pdb; pdb.set_trace()
	q_desired =  desired_kinematics[step_number, np.ix_([0,3])][0]
	q_dot_desired = desired_kinematics[step_number, np.ix_([1,4])][0]
	q_error = q_desired - real_attempt_positions[step_number-1,:]
	q_dot_in = q_dot_desired + K*q_error
	q_double_dot_in = [np.gradient(desired_kinematics[step_number-2:step_number+1,1],edge_order=1)[-1]/timestep, np.gradient(desired_kinematics[step_number-2:step_number+1,4],edge_order=1)[-1]/timestep]#desired_kinematics[step_number, np.ix_([2,5])][0]#
	#import pdb; pdb.set_trace()
	desired_kinematics = [q_desired[0], q_dot_in[0], q_double_dot_in[0], q_desired[1], q_dot_in[1], q_double_dot_in[1]]
	return desired_kinematics

def close_loop_run_fcn(model, desired_kinematics, K, plot_outputs=True, Mj_render=False, chassis_fix=True, timestep=.005):
	if chassis_fix:
		Mj_model = load_model_from_path("./models/nmi_leg_w_chassis_fixed.xml")
	else:
		Mj_model = load_model_from_path("./models/nmi_leg_w_chassis_walk.xml")
	sim = MjSim(Mj_model)
	if Mj_render:
		viewer = MjViewer(sim)
	sim_state = sim.get_state()
	control_vector_length=sim.data.ctrl.__len__()
	print("control_vector_length: "+str(control_vector_length))
	est_activations = estimate_activations_fcn(model, desired_kinematics)
	#[real_attempt_kinematics, real_attempt_activations, chassis_pos] = run_activations_fcn(est_activations, chassis_fix=True, timestep=0.005, Mj_render=Mj_render)
	number_of_task_samples = desired_kinematics.shape[0]
	chassis_pos=np.zeros(number_of_task_samples,)
	input_kinematics = np.zeros(desired_kinematics.shape)
	real_attempt_positions = np.zeros([number_of_task_samples,2])
	real_attempt_activations = np.zeros([number_of_task_samples,3])
	sim.set_state(sim_state)
	current_positions_array = sim.data.qpos[-2:]
	for ii in range(number_of_task_samples):
		if ii < 2:
			print(ii)
			input_kinematics[ii,:] = desired_kinematics[ii,:]
		else:
			input_kinematics[ii,:] = calculate_inputkinematics(
				step_number=ii,
				real_attempt_positions=real_attempt_positions,
				desired_kinematics=desired_kinematics,
				K=K,
				timestep=timestep)
		est_activations[ii,:] = model.predict([input_kinematics[ii,:]])[0,:]
		#import pdb; pdb.set_trace()
		sim.data.ctrl[:] = est_activations[ii,:]
		sim.step()
		previous_positions_array = current_positions_array
		current_positions_array = sim.data.qpos[-2:]
		current_desired_velocity_array = input_kinematics[ii, np.ix_([1,4])][0,:]#(current_positions_array - current_positions_array)/timestep
		chassis_pos[ii]=sim.data.get_geom_xpos("Chassis_frame")[0]
		#import pdb; pdb.set_trace()
		real_attempt_positions[ii,:] = current_positions_array
		real_attempt_activations[ii,:] = sim.data.ctrl
		if Mj_render:
			viewer.render()
	real_attempt_kinematics = positions_to_kinematics_fcn(
		real_attempt_positions[:,0], real_attempt_positions[:,1], timestep=timestep)
	error0 = error_cal_fcn(desired_kinematics[:,0], real_attempt_kinematics[:,0])
	error1 = error_cal_fcn(desired_kinematics[:,3], real_attempt_kinematics[:,3])
	average_error = 0.5*(error0+error1)
	if plot_outputs:
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.plot(range(desired_kinematics.shape[0]), desired_kinematics[:,0], range(desired_kinematics.shape[0]), real_attempt_kinematics[:,0])
		plt.ylabel("q0 desired vs. simulated")
		plt.subplot(2, 1, 2)
		plt.plot(range(desired_kinematics.shape[0]), desired_kinematics[:,3], range(desired_kinematics.shape[0]), real_attempt_kinematics[:,3])
		plt.ylabel("q1  desired vs. simulated")
		plt.xlabel("Sample #")
		plt.show(block=True)
	return average_error

simplefilter(action='ignore', category=FutureWarning)

# [babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
# model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)
# cum_kinematics = babbling_kinematics
# cum_activations = babbling_activations


# np.random.seed(0)
# pickle.dump([model,cum_kinematics, cum_activations],open("results/mlp_model.sav", 'wb'))

[model,cum_kinematics, cum_activations] = pickle.load(open("results/mlp_model.sav", 'rb')) # loading the model

desired_kinematics = create_sin_cos_kinematics_fcn(attempt_length = 10 , number_of_cycles = 7, timestep = 0.005)
average_error = open_loop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=True, Mj_render=False)
print("average open-loop error is: ", average_error)
average_error = close_loop_run_fcn(model=model, desired_kinematics=desired_kinematics, K=[2, 20], plot_outputs=True, Mj_render=False)
print("average close-loop error is: ", average_error)
#import pdb; pdb.set_trace()



