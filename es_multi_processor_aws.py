import numpy as np
#import matplotlib.pyplot as plt
from battle_env_aws import game_map,nn
from PIL import Image
#import imageio
from joblib import Parallel, delayed


config = {
	'l' : 500,                                  # length of the image
	'w' : 500,                                  # breadth of the image
	'team_split' : [2,2],                       # 2 vs 2 game
	'rmax' : np.pi/10,                          # rotation quanta. each rotate action will rotate by rmax.
	'vmax' : 25,                                # maximum velocity in a given direction per timestep
	'mv' : 35,                                  # missile velocity per timestep
	'ship_r' : 20,                              # 10% of the min(L,W) # ship radius in pixels
	'm_r' : 4,                                  # 4% of min(L,W) # missile radius in pixels
	'types' : 3,                                # only attack ships , 0: attack, 1: refuel, 2: medic
	'health' : 255,                             # maximum health of a ship
	'fuel' : 1000,                              # maximum fuel of a ship
	'mhloss': 5,                                # missile hit health loss
	'shloss':10,                                # ship hit health loss
	'floss': 5,                                 # fuel consumption per timestep
	'refuel' : 20,                              # refuel rate per timestep
	'max_ships': 20,                            # maximum number of ships allowed
	'color_list':['y','orange','pink','black'], # colors for upto four teams, if more teams are present add more colors.
	'gamma':0.9,                                # RL parameters
	'lambda': 1,                                # RL parameters, TD lambda method
	'batch' : 16,                               # batch size 
	'timesteps':128,                            # for LSTM
	'ent_coef':0.01,                            # entropy coefficient
	'vf_coef':0.1                               # value loss coefficient 
}

#X = env.get_state()

# write run episode, with n frames from previous steps
def run_episode(env,agent,w1,w2,record = False,frameskip = 4):
	#records 
	# this run episode function is only for 2 vs 2 game with attack ships, modify for team splits

	env = game_map(config = config)

	env.add_ship(0,0,0)
	env.add_ship(0,0,1)
	env.add_ship(1,0,0)
	env.add_ship(1,0,1)

	frames = []
	frame_vec = []
	num_frames = 4 # can go in config
	state = np.zeros([4,24]) # for 2 vs 2 team with 24 vector state space
	k = 3
	X = env.get_vec()
	agent[0].set_weights(w = w1)
	agent[1].set_weights(w = w2)
	for i in range(1000):
		state[k,:] = X
		#action = np.random.randint(0,4)
		team = np.random.randint(0,2) 
		action = agent[team].action(state.flatten())
		agent_id = np.random.randint(0,2)
		env.step(team,0,agent_id,action)
		X = env.get_vec()
		k-=1
		if k==-1:
			k=3
		if record==True and i%4==0:
			frames.append(env.get_state())
			frame_vec.append(state.flatten())
	rew_arr = []
	for i in range(len(env.team_split)):
		rew_arr.append(env.get_rew(i))
	#return frames,frame_vec,rew_arr  

	return rew_arr  

def main(): 

	env = game_map(config = config)

	env.add_ship(0,0,0)
	env.add_ship(0,0,1)
	env.add_ship(1,0,0)
	env.add_ship(1,0,1)

	agent = [nn(state_space=4*24,action_space=4,layers=[]) for i in range(len(config['team_split']))]

	nsample = 300
	sigma = 0.1
	alpha = 0.1
	niter = 50000
	agent = [nn(state_space=4*24,action_space=4,layers=[]) for i in range(len(config['team_split']))]
	total_elem = agent[0].total_elem
	w = np.random.randn(total_elem) # initial guess
	w_hist = []
	ncores = 16
	rew = []
	for i in range(niter):
		print("iteration {} : {}".format(i,niter))
		rew = np.zeros(nsample)
		N = np.random.randn(nsample,total_elem)  
		
		if i<40:
			sel = -1
			w_hist.append(w)
		else:
			sel = np.random.randint(0,40)
			w_hist.append(w)
			w_hist.pop(0)
		 
		rew_arr = Parallel(n_jobs = ncores)(delayed(run_episode)(env = env,agent = agent,w1=w +sigma*N[j],w2= w_hist[sel]+ sigma*N[j],record = False) for j in range(nsample))
		rew = np.array(rew_arr)[:,0]
		np.save("weights_best_niter_{}_1.npy".format(i),w + sigma*N[np.argmax(rew)])
		np.save("weights_best_niter_{}_2.npy".format(i),w_hist[sel] + sigma*N[np.argmax(rew)])
		print(rew)
		print("maximum reward is {}".format(np.max(rew)))
		if np.std(rew)!=0:
			rew = (rew - np.mean(rew))/np.std(rew)
		else:
			rew = (rew - np.mean(rew))
		
		w = w + alpha/(nsample*sigma) * np.dot(N.T, rew)

		rew1 = run_episode(env = env, agent = agent, w1 = w , w2 = w_hist[sel] + sigma*N[0])
		print("reward is {}".format(rew1))

		np.save("w{}.npy".format(i),w)

	np.save("w.npy",w)

	return

if __name__ == '__main__':
	main()
		
