"""
Akhilesh Khope
PhD candidate
Electrical engineering
UC Santa Barbara



The following code implements a many on many environment for fights between space ships. 
There can be multiple teams with multiple spaceships and space ships in each team are identified by their unique color.
All space ships can be controlled with a neural network which can read state give by the screen pixels or 
a vector that contains health and ammunition status of its team members and the same for opponent teams.
There can be three types of ships in each team: attack,refuel and medic ship. 
All three ships can attack and have finite fuel, health and missiles, 
but the refuel ship can refuel other ships in its vicinity at a fixed fuel injection rate. 
The medic ship can repair damages, i.e. increase health at a fixed rate. Each team gets at most one medic and refuel ship.
The ships can damage each other, by either firiing missiles or kamikaze (bang each other).
Some interesting behaviours I hope to observe are :
1. attack ships shielding the medic or refuel ships by surrounding them and moving together.
2. some ships diverting attention and others flanking
3. many more interesting battle strategies.

Instead of having one centralized controller for the team its interesting to give seperate brain or
a neural network to each ship as the dynamics can demonstrate a pack mentality. 
Some ships sacrificing themselves for the team, on the fly assigning roles like attacking, defending, and 
surrounding the other team or clustering together. 

This version implements only attack ships.
Each space ship has following controls:
1. turn clockwise
2. turn anticlockwise
3. move
4. fire 
5. hold, i.e do nothing

an example configuration file that can set the environment is given below 
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

each space ship from each team can act every timestep. And teams are chosen at random.

a team looses when all the space ships from a particular team are distroyed.


"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
import scipy.signal


class game_map:
    def __init__(self,config):
        self.team_split = np.array(config['team_split'])
        self.types = config['types']
        self.map_dict = {}
        # dictionary of teams
        for i in range(len(self.team_split)):
            self.map_dict[i] = {}
            for j in range(self.types):
                self.map_dict[i][j]={}
        
        self.l = config['l']
        self.w = config['w']
        self.rmax = config['rmax']
        self.vmax = config['vmax']
        self.mv = config['mv']
        self.ship_r = config['ship_r']
        self.m_r = config['m_r']
        self.max_health = config['health']
        self.fuel = config['fuel']
        self.mhloss = config['mhloss']
        self.shloss = config['shloss']
        self.floss = config['floss']
        self.refuel = config['refuel']
        
        self.num_miss = 0 # number of missiles fired
        self.num_ships = 0 # number of ships initialized
        self.max_missiles = 1000
        self.max_ships = config['max_ships']
        
        self.init_x = np.zeros(self.max_ships)
        self.init_y = np.zeros(self.max_ships)
        self.missiles = {}
        
        self.color_list = config['color_list']
        
        #termination condition
        self.done = False


        
        self.dead_ships = {}
        
        
        #initialized an empty game_map
        # class functions:
        # add_ship(team,ship_type,ship_id) ... 
        # get_state() ... return image
        # step(team,ship_type,ship_id,action) .. updates state for the game_map
        # action .. for attack ship turn,move,fire,hold
        # action .. for refuel ship turn,move,fire,hold
        # action .. for medic ship turn,move,fire,hold
    def add_ship(self,team,ship_type,ship_id):
        # adds a ship to a random location
        self.map_dict[team][ship_type][ship_id] = {}
        if self.num_ships==0:
            self.map_dict[team][ship_type][ship_id]['x'] = np.random.uniform(self.ship_r,self.l-self.ship_r)
            self.map_dict[team][ship_type][ship_id]['y'] = np.random.uniform(self.ship_r,self.w-self.ship_r)
            self.init_x[self.num_ships] = self.map_dict[team][ship_type][ship_id]['x'] 
            self.init_y[self.num_ships] = self.map_dict[team][ship_type][ship_id]['y'] 
            self.map_dict[team][ship_type][ship_id]['theta'] = np.random.uniform(0,2*np.pi)
            self.map_dict[team][ship_type][ship_id]['missiles'] = self.max_missiles
            self.map_dict[team][ship_type][ship_id]['fuel'] = self.fuel
            self.map_dict[team][ship_type][ship_id]['health'] = self.max_health
            self.num_ships+=1
        else:            
            flag = False
            while True:
                temp_x = np.random.uniform(self.ship_r,self.l-self.ship_r)
                temp_y = np.random.uniform(self.ship_r,self.w-self.ship_r)
                for i in range(self.max_ships):
                    if self.init_x[i]==0: # this implies that no ship is present, since init_x is initialized to zero
                        break
                    if np.linalg.norm(np.array([temp_x,temp_y])-np.array([self.init_x[i],self.init_y[i]])) > 2*self.ship_r:
                        flag = True
                    if flag==True:
                        break
                if flag==True:
                    break
            self.map_dict[team][ship_type][ship_id]['x'] = temp_x
            self.map_dict[team][ship_type][ship_id]['y'] = temp_y
            self.init_x[self.num_ships] = self.map_dict[team][ship_type][ship_id]['x'] 
            self.init_y[self.num_ships] = self.map_dict[team][ship_type][ship_id]['y']
            self.map_dict[team][ship_type][ship_id]['theta'] = np.random.choice(np.arange(0,2*np.pi,self.rmax))
            self.map_dict[team][ship_type][ship_id]['missiles'] = self.max_missiles
            self.map_dict[team][ship_type][ship_id]['fuel'] = self.fuel
            self.map_dict[team][ship_type][ship_id]['health'] = self.max_health
            self.num_ships+=1
        
        return
    
    def step(self,team,ship_type,ship_id,action):
        # step(team,ship_type,ship_id,action) .. updates state for the game_map
        # action .. for attack ship turn,move,fire,hold
        # action .. for refuel ship turn,move,fire,hold
        # action .. for medic ship turn,move,fire,hold
        
        # ship movement and missile fire in the enviroment
        if self.map_dict[team][ship_type][ship_id]['health'] >0:
            if action == 0:
                # turn counterclockwise
                self.map_dict[team][ship_type][ship_id]['theta']+=self.rmax
                self.map_dict[team][ship_type][ship_id]['theta']%=(2*np.pi)
            if action == 1:
                # turn clockwise
                self.map_dict[team][ship_type][ship_id]['theta']-=self.rmax
                self.map_dict[team][ship_type][ship_id]['theta']%=(2*np.pi)
            if action == 2:
                # move
                theta = self.map_dict[team][ship_type][ship_id]['theta']
                temp_x = self.map_dict[team][ship_type][ship_id]['x'] + self.vmax*np.cos(theta)
                temp_y = self.map_dict[team][ship_type][ship_id]['y'] + self.vmax*np.sin(theta)
                if temp_x<(self.l-self.ship_r) and temp_x > self.ship_r and temp_y > self.ship_r and temp_y < (self.w - self.ship_r):
                    self.map_dict[team][ship_type][ship_id]['x'] = temp_x
                    self.map_dict[team][ship_type][ship_id]['y'] = temp_y
            if action == 3:
                # fire
                if self.map_dict[team][ship_type][ship_id]['missiles'] >0:
                    #print('fire')
                    self.missiles[self.num_miss] = {}
                    #print(self.missiles)
                    theta = self.map_dict[team][ship_type][ship_id]['theta']
                    temp_x = self.map_dict[team][ship_type][ship_id]['x'] + (self.m_r)*np.cos(theta)
                    temp_y = self.map_dict[team][ship_type][ship_id]['y'] + (self.m_r)*np.sin(theta)
                    self.missiles[self.num_miss]['x'] = temp_x
                    self.missiles[self.num_miss]['y'] = temp_y
                    self.missiles[self.num_miss]['theta'] = theta
                    self.missiles[self.num_miss]['fired_by'] = ship_id
                    self.map_dict[team][ship_type][ship_id]['missiles']-=1
                    self.num_miss+=1
                    #print(self.missiles)
            # action ==4 is hold
                    
        # also calculating health losses from ships overlapping 
        self.dead_missiles = {}
        
        for i in self.missiles:
            #propagate all missiles
            #if i!=self.num_miss:
            #print('propagate')
            self.missiles[i]['x']+= self.mv*np.cos(self.missiles[i]['theta'])
            self.missiles[i]['y']+= self.mv*np.sin(self.missiles[i]['theta'])
        
        for i in self.missiles:
            mis_coord = np.array([self.missiles[i]['x'],self.missiles[i]['y']])
            #check if going out of bounds
            if (self.missiles[i]['x'] - self.m_r < 0) or (self.missiles[i]['x'] - self.l + self.m_r > 0) or (self.missiles[i]['y'] - self.m_r < 0) or (self.missiles[i]['y'] - self.w + self.m_r > 0 ): 
                self.dead_missiles[i] = 0
                #print(1)

            # check for collision with other missiles
            for j in self.missiles:
                if i!=j and (j not in self.dead_missiles) and (i not in self.dead_missiles):
                    mis_coord2 = np.array([self.missiles[j]['x'],self.missiles[j]['y']])
                    if np.linalg.norm(mis_coord-mis_coord2) < 2*(self.m_r):
                        #print('missile collision')
                        #print(self.m_r)
                        #print(np.linalg.norm(mis_coord-mis_coord2))
                        #del self.missiles[i] # missile destroyed
                        self.dead_missiles[i] = 0
                        #del self.missiles[j] # missile destroyed
                        self.dead_missiles[j] = 0



            # check if colliding with any ships

            for l in self.map_dict:
                for j in self.map_dict[l]:
                    for k in self.map_dict[l][j]:
                        if (i not in self.dead_missiles) and (k not in self.dead_ships) and k!=self.missiles[i]['fired_by']:
                            ship_coord = np.array([self.map_dict[l][j][k]['x'],self.map_dict[l][j][k]['y']])
                            if np.linalg.norm(mis_coord-ship_coord) <(self.ship_r + self.m_r):
                                self.map_dict[l][j][k]['health']-=self.mhloss
                                #print(np.linalg.norm(mis_coord-ship_coord))
                                #del self.missiles[i] # missile destroyed
                                self.dead_missiles[i] = 0
                                if self.map_dict[l][j][k]['health'] <=0:
                                    # ship destroyed
                                    #del self.map_dict[l][j][k]
                                    self.dead_ships[k] = [l,j]

        # check if ships coliding with each other

        for l in self.map_dict:
                for j in self.map_dict[l]:
                    for k in self.map_dict[l][j]:
                        for m in self.map_dict[l][j]:
                            if k!=m and (k not in self.dead_ships) and (m not in self.dead_ships):
                                ship_coord1 = np.array([self.map_dict[l][j][k]['x'],self.map_dict[l][j][k]['y']])
                                ship_coord2 = np.array([self.map_dict[l][j][m]['x'],self.map_dict[l][j][m]['y']])
                                if np.linalg.norm(ship_coord1-ship_coord2) < 2*(self.ship_r):
                                    self.map_dict[l][j][k]['health']-=self.shloss
                                    self.map_dict[l][j][m]['health']-=self.shloss
                                if self.map_dict[l][j][k]['health'] <=0:
                                    # ship destroyed
                                    #del self.map_dict[l][j][k]
                                    self.dead_ships[k] = [l,j]
                                    break # breaks out of outer for loop with iteration over k
                                if self.map_dict[l][j][m]['health'] <=0:
                                    # ship destroyed
                                    #del self.map_dict[l][j][m]
                                    self.dead_ships[m] = [l,j]
        #delete the dead missiles and dead ships from the respective dictionaries
        for i in self.dead_missiles:
            del self.missiles[i]
        
        
        # for i in self.dead_ships:
        #     del self.map_dict[dead_ships[i][0]][dead_ships[i][1]][i]

        #print(dead_missiles)
        #print(dead_ships)
                
        return self.get_state(),self.get_rew(team),self.termi_cond()
    
    def get_state(self):
        my_dpi = 80
        fig = plt.figure(figsize=(500/my_dpi, 500/my_dpi), dpi=my_dpi)
        canvas = plt.get_current_fig_manager().canvas
        #fig.axes()
        scaling = self.l
        for i in self.map_dict:
            for j in self.map_dict[i]:
                for k in self.map_dict[i][j]:  
                    temp_x = self.map_dict[i][j][k]['x']
                    temp_y = self.map_dict[i][j][k]['y']
                    temp_theta = self.map_dict[i][j][k]['theta']
                    
                    circle = plt.Circle((temp_x, temp_y), radius=self.ship_r, fc=self.color_list[i])
                    plt.gca().add_patch(circle)
                    direction = plt.Arrow(temp_x, temp_y,self.ship_r*np.cos(temp_theta),self.ship_r*np.sin(temp_theta), width =4)
                    plt.gca().add_patch(direction) 
                    
        for i in self.missiles:
            circle = plt.Circle((self.missiles[i]['x'], self.missiles[i]['y']), radius=self.m_r, fc='r')
            plt.gca().add_patch(circle)

        plt.xlim([0,self.l])
        plt.ylim([0,self.w])
        #plt.axis('scaled')
        
        #canvas = plt.get_current_fig_manager().canvas

        agg = canvas.switch_backends(FigureCanvasAgg)
        agg.draw()
        s = agg.tostring_rgb()

        # get the width and the height to resize the matrix
        l, b, w, h = agg.figure.bbox.bounds
        w, h = int(w), int(h)

        X = np.fromstring(s, np.uint8)
        X.shape = h, w, 3

        plt.close()

        return X    
    def termi_cond(self):
        count = np.zeros(len(self.team_split))
        for i in self.map_dict:
            for j in self.map_dict[i]:
                for k in self.map_dict[i][j]:
                    if self.map_dict[i][j][k]['health']<=0:
                        count[i]+=1
        
        
        mask = count<=self.team_split
        
        if np.sum(mask)==(len(self.team_split)-1):
            self.done = True
        
        
        self.winner = np.argmin(mask)
    
        return self.done
    def get_info(self,team,ship_type,ship_id):
        
        return self.map_dict[team][ship_type][ship_id]
    
    def get_rew(self,team):
        # only type zero ships considered
        k=0
        rew=0
        for i in range(len(self.team_split)):
            for j in range(self.team_split[i]):
                if k not in self.dead_ships:
                    if i==team:
                        rew+=self.map_dict[i][0][k]['health']
                    else:
                        rew-=self.map_dict[i][0][k]['health']
                    k+=1
        return rew
        