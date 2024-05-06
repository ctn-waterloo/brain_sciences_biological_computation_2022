import nengo
import numpy as np

    
## TD($\lambda$) Learning Rule
class ActorCriticTDLambda(nengo.processes.Process):
    '''Create nengo Node with input, output and state.
    This node is where the TD(lambda) learning rule is applied.
    
    Inputs: state, reward, chosen action, whether or not the environment was reset
    Outputs: updated state value, updated action values
    '''
    def __init__(self, n_actions, alpha=0.1, beta=0.85, gamma=0.9, lambd=0.9, env_dt=None):
        self.n_actions = n_actions ##number of possible actions for action-value space
        self.alpha = alpha ##learning rate
        self.beta = beta ##discount factor for action values
        self.gamma = gamma ##discount factor for state values
        self.lambd = lambd ##discount factor for eligibility traces
        self.env_dt = env_dt ##number of environment steps per simulation step
        
        ## When using a different time step duration for the environment, scale alpha
        if self.env_dt != None:
            self.alpha = alpha/(self.env_dt/0.001)
            
        ## Input = reward + state representation + action values for current state
        ## Output = action values for current state + state value
        super().__init__(default_size_in=n_actions + 2, default_size_out=n_actions + 1) 
        
    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        '''Get a dictionary of signals to represent the state of this process.
        This will include: the representation of the previous state (prev_state_rep),
        the eligibility traces for the state (trace) and the actions (action_trace),
        the initial value of the previous state (0) (prev_value), and the weight matrix/look-up table (w).
        Weight matrix shape = [(n_actions + state) * size of state representation output]'''
        
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        ##return the state dictionary
        return dict(prev_state_rep=np.zeros(dim),
                    trace=np.zeros(dim),
                    action_trace=np.zeros((self.n_actions, dim)),
                    prev_value=0,
                    w=np.zeros((self.n_actions+1, dim)))
    
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        '''Function that advances the process forward one time step.'''
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        
        dim = shape_in[0]-2-self.n_actions
        
        def step_TDlambda(t, x, state=state):
            ''' Function for performing TD(lambda) update at each timestep.
            Inputs: state representation, chosen action, reward
                and whether or not the env was reset
            Params:
                t = time
                x = inputs 
                state = the dictionary created in make_state.
            Outputs: updated state values, updated action values'''

            current_state_rep = x[:dim] ##get the state representation of current state
            prev_state_rep = state['prev_state_rep'] ##get representation of the previous state
            prev_action = x[dim:-2] ##get the action that was taken
            reward = x[-2] ##get reward received following action
            reset = x[-1] ##get whether or not env was reset

            ##get the dot product of the weight matrix and state representation
            ##results in 1 value for the state, and 1 value for each action
            result_values = state['w'].dot(current_state_rep)

            ##get the value of the current state
            current_state_value = result_values[0]

            ##Do the TD(lambda) update
            ##skip if the environment has just been reset
            if not reset:
                ##calculate the td error term
                td_error = reward + (self.gamma*current_state_value) - state['prev_value']

                ##calculate a scaling factor
                ##this scaling factor allows us to switch from updating
                ##a look-up table to updating weights 
                scale = np.sum(prev_state_rep**2)
                if scale != 0:
                    scale = 1.0 / scale

                ##update the state and action eligibility traces
                state['trace'] *= self.gamma * self.lambd
                state['action_trace'] *= self.gamma * self.lambd
                ##Accummulative trace - increment the state eligibility trace by the previous state representation
                state['trace'] += prev_state_rep * scale
                ##increment the action eligibility trace for the previous action by the previous state representation
                state['action_trace'] += np.outer(prev_action, prev_state_rep * scale)
                #state['action_trace'][:,prev_action] += prev_state_rep * scale

                ##update the weights for the state value
                state['w'][0] += self.alpha*td_error*state['trace']

                ##update the weights for the action values
                ##multiply the entire state represention by (action value * beta * tderror)
                ##scale these values and then update the weight matrix/look-up table
                dw = state['action_trace'] * self.beta * td_error
                state['w'][1:] += dw

                ##Uncomment these to compare TD(0) result to TD(lambda) where lambda = 0 (should be the same)
                #dw2 = np.outer(prev_action*self.beta*td_error, prev_state_rep)*scale
                #assert np.allclose(dw, dw2)

            ##change the 'prev_state_rep' to the representation of the current state in this trial
            state['prev_state_rep'][:] = current_state_rep
            ##change the 'prev_value' to the value of the current state in this trial
            state['prev_value'] = result_values[0]

            ##return updated state value for update state and action values for current state
            return result_values          
        return step_TDlambda