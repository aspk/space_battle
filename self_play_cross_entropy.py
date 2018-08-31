"""
Akhilesh Khope
PhD candidate
Electrical Engineering
UC Santa Barbara

akhileshsk@gmail.com
"""
import numpy as np
#import matplotlib.pyplot as plt
from battle_env_aws import game_map,nn
from PIL import Image
#import imageio
import time



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
	for i in range(200):
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

# runs an iteration of cross entropy simulation
def run_iter(iter_num):
    navg = 20
    nsamples = 100
    nelite = 10
    mean = np.random.normal(size = total_elem)
    std = np.random.uniform(0,1,size = total_elem)

    noise = 5
    niter = 50
    rew_progress = []
    mean_hist = []
    std_hist = []
    look_back = 20
    mean_hist.append(mean)
    std_hist.append(std)
    for i in range(niter):
        if i%10 == 0 and i!=0:
            print("iteration {} : {} : {} ".format(iter_num,i,niter))
            print(rew_progress[-1])
        rew_arr = []

        w1_arr = np.random.normal(mean,std,size = [nsamples,total_elem])
        if i>look_back:
            random_lb = np.random.randint(0,look_back)
        elif i==0:
            random_lb = 0
        else:
            random_lb = np.random.randint(0,i)
        w2_arr = np.random.normal(mean_hist[random_lb],std_hist[random_lb],size = [navg,total_elem])
        for w1 in w1_arr:
            rew = 0
            for w2 in w2_arr:
                rew += run_episode(env = env,agent = agent,w1 = w1,w2 = w2,record =False)[0]

            rew_arr.append(np.mean(rew))


        elite_ind = np.argsort(rew_arr)[-nelite:]
        w_elite = w1_arr[elite_ind]
        mean = np.mean(w_elite,0)
        std = np.std(w_elite,0)
        std+=noise*np.ones(total_elem)/(i+1)
        rew_progress.append(np.mean(rew_arr[-nelite:]))
        #print(rew_progress[-1])
        mean_hist.append(mean)
        std_hist.append(std)
        if len(mean_hist)>look_back:
            mean_hist.pop(0)
            std_hist.pop(0)
        
    np.save('w_elite_{}.npy'.format(iter_num),np.array(w_elite))
    np.save('mean_hist_{}.npy'.format(iter_num),np.array(mean_hist))
    np.save('std_hist_{}.npy'.format(iter_num),np.array(std_hist))
        
    return
        
def main():
	#generate hundred agents for 2 vs 2 play
	for i in range(0,100):
	    start_time = time.time()
	    print("bot number {}".format(i))
	    run_iter(i)
	    end_time = time.time()
	    print("time required is {}".format(end_time-start_time))
	return

if __name__ == '__main__':
	main()