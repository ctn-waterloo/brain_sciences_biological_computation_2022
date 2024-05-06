import nengo
import numpy as np

## TD(0) Learning Rule ##
class ActorCriticTD0(nengo.processes.Process):
    '''Create nengo Node with input, output and state.
    This node is where the TD(0) learning rule is applied.
    
    Inputs: state, reward, chosen action, whether or not the environment was reset
    Outputs: updated state value, updated action values
    '''
    def __init__(self, n_actions, alpha = 0.1, beta = 0.9, gamma = 0.95, n=None, lambd=None, env_dt=None):
        self.n_actions = n_actions ##number of possible actions for action-value space
        self.alpha = alpha ##learning rate
        self.beta = beta ##discount factor for action values
        self.gamma = gamma ##discount factor for state values
        self.n = n ##value of n - the number of steps between the current state and the state being updated
        self.lambd = lambd ##discount factor for eligibility traces
        self.env_dt = env_dt ##number of environment steps per simulation step
        
        if self.env_dt != None:
            self.alpha = alpha/(self.env_dt/0.001)
        
        ## Input = reward + state representation + action values for current state
        ## Output = action values for current state + state value
        super().__init__(default_size_in=n_actions + 2, default_size_out=n_actions + 1) 
        
    def make_state(self, shape_in, shape_out, dt, dtype=None):
        '''Get a dictionary of signals to represent the state of this process.
        This will include: the representation of the state being updated (update_state_rep),
        the initial value of the state (0) (update_value), and the weight matrix/look-up table (w).
        Weight matrix shape = [(n_actions + state) * size of state representation output]'''
        
        ## set dim = length of each row in matrix/look-up table 
        ## where a nengo ensemble is used to store the state representation, 
        ## this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        
        ## return the state dictionary
        return dict(update_state_rep=np.zeros(dim),
                    update_value=0,
                    w=np.zeros((self.n_actions+1, dim)))
    
    def make_step(self, shape_in, shape_out, dt, rng, state):
        '''Create function that advances the process forward one time step.'''
        ## set dim = length of each row in matrix/look-up table 
        ## where a nengo ensemble is used to store the state representation, 
        ## this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        
        ## One time step
        def step_TD0(t, x, state=state):
            ''' Function for performing TD(0) update at each timestep.
            Inputs: state representation, chosen action, reward
                and whether or not the env was reset
            Params:
                t = time
                x = inputs 
                state = the dictionary created in make_state.
            Outputs: updated state value, updated action values'''

            current_state_rep = x[:dim] ##get the state representation of current state
            update_state_rep = state['update_state_rep'] ##get representation of state being updated
            update_action = x[dim:-2] ##get the action being updated (i.e. the action that was taken)
            reward = x[-2] ##get reward received following action
            reset = x[-1] ##get whether or not env was reset

            ## get the dot product of the weight matrix and state representation
            ## results in 1 value for the state, and 1 value for each action
            result_values = state['w'].dot(current_state_rep)

            ## get the value of the current state
            current_state_value = result_values[0]


            ## Do the TD(0) update
            if not reset: ##skip this if the env has just been reset
                ## calculate td error term
                td_error = reward + (self.gamma*current_state_value) - state['update_value']

                ## calculate a scaling factor
                ## this scaling factor allows us to switch from updating
                ## a look-up table to updating weights 
                scale = np.sum(update_state_rep**2)
                if scale != 0:
                    scale = 1.0 / scale

                ## update the state value
                state['w'][0] += self.alpha*td_error*update_state_rep*scale
                ## update the action values
                ## multiply the entire state represention by (action value * beta * tderror)
                ## scale these values and then update the weight matrix/look-up table
                dw = np.outer(update_action*self.beta*td_error, update_state_rep)
                state['w'][1:] += dw*scale

            ## calculate the updated value for update state and add it to the result_values array
            result_values[0] = state['w'].dot(state['update_state_rep'][:])[0]
            ## change the state to be updated to the current state in this step    
            state['update_state_rep'][:] = current_state_rep
            ## change the value being updated to the value of the current state in this step
            state['update_value'] = current_state_value

            ## return updated state value for update state and action values for current state
            return result_values             
        return step_TD0
    