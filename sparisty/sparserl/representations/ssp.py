import numpy as np
import nengo_spa
import math
import gymnasium as gym
import minigrid
from sparserl.sspspace import HexagonalSSPSpace, RandomSSPSpace


class SSPRep(object):
    '''Create state representation with grid-cells'''
    
    def __init__(self, N, n_scales=8, n_rotates=4, ssp_dim=193, hex=True, length_scale=1):
        if hex:
            self.ssp_space = HexagonalSSPSpace(N,
                                   domain_bounds=None, scale_min=0.5, 
                                   scale_max = 2, n_scales=n_scales, n_rotates=n_rotates,
                                   length_scale=length_scale)
        else:
            self.ssp_space = RandomSSPSpace(N,ssp_dim = ssp_dim,
                                   domain_bounds=None, 
                                    length_scale=length_scale)
        self.size_out = self.ssp_space.ssp_dim
       
    
    def map(self, state):   
        ssppos = self.ssp_space.encode(state)
        return ssppos.reshape(-1)

    def make_encoders(self, n_neurons):
        return self.ssp_space.sample_grid_encoders(n_neurons)
    
    def get_state(self, state, env):
        return state