import minigrid
from minigrid.wrappers import *
import gymnasium as gym
from scipy.spatial import distance
import matplotlib.pyplot as plt

class MiniGrid_wrapper(gym.Env):
    '''Wrapper for the mini-grid environment.
    This allows us to access the agent's state in the same way as we would with 
    other gym environments.
    '''
    def __init__(self, env, dist=None):
        self.dist = dist
        self.orig_env = gym.make(env, render_mode = 'rgb_array') #call minigrid environment
        self.action_space = spaces.Discrete(3) #self.orig_env.action_space #call minigrid actionspace
        #self.observation_space = [8,8,4] #set observation space
        self.orig_env.reset() #reset environment
        
        self.min_x_pos = 0
        self.max_x_pos = self.orig_env.width-1
        self.min_y_pos = 0
        self.max_y_pos = self.orig_env.height-1
        self.min_dir = 0
        self.max_dir = 3

        self.low = np.array([self.min_x_pos, self.min_y_pos, self.min_dir], dtype=np.float32)
        self.high = np.array([self.max_x_pos, self.max_y_pos, self.max_dir], dtype=np.float32)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
    def render(self):
        #plt.imshow(orig_env.render('rgb_array'))
        return plt.imshow(self.orig_env.render())

    def reset(self):
        obs = self.orig_env.reset() #reset environment
        #get agent's state (x position, y position, direction)
        pos = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
        return pos #return state
    
    def get_goal(self):
        for grid in self.orig_env.grid.grid:
            if grid is not None and grid.type == "goal":
                return(grid.cur_pos)
        
    def step(self, a):
        obs, reward, terminated, truncated, info = self.orig_env.step(a) #do step in environment
        done = terminated or truncated
        #get agent's state (x position, y position, direction)
        pos = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
            
        if self.dist == 'Euclidean':
            ## calculate euclidean distance from goal
            goal = self.get_goal()
            d = math.sqrt(((goal[0] - pos[0])**2) + ((goal[1] - pos[1])**2))

            ## calculate reward based on distance
            if done == False:
                reward = (1-(d*0.1))*0.01
                
        elif self.dist == 'CityBlock':
            ## calculate manhattan distance from goal
            goal = self.get_goal()
            d = distance.cityblock(pos[:2], goal)

            ## calculate reward based on distance
            if done == False:
                reward = (1-(d*0.1))*0.01
                
        #goal = self.get_goal()
        #if obs[:2] == goal:
        #    reward = 1
            
        #return state, reward, done and info
        return pos, reward, done, info