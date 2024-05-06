from tqdm import tqdm #progress bar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import nengo
import pytry
import sparserl
from sparserl.utils import softmax

class ActorCriticLearn(pytry.Trial):
    ## PARAMETERS ##
    def params(self):
        ## Task
        self.param('gym environment: MiniGrid', env='MiniGrid-Empty-8x8-v0')
        self.param('duration of time step in gym environment', env_dt=None)
        self.param('number of learning trials', trials=1000)
        self.param('length of each trial', steps=200)
        
        ## Representation
        self.param('method of calculating distance when using dynamic reward', dist=None)
        self.param('representation', rep=sparserl.representations.SSPRep(3)),
        
        ## Learning Rule 
        self.param('select learning rule', rule=sparserl.rules.ActorCriticTD0),
        self.param('learning rate', lr=0.1)
        self.param('action discount factor', act_discount=0.85)
        self.param('value discount factor', state_discount=0.9)
        self.param('n for TD(n)', n=None)
        self.param('lambda value', lambd=None)
        
        ## Ensemble Parameters 
        self.param('number of neurons', n_neurons=None)
        self.param('neuron sparsity', sparsity=None)
        self.param('ensemble neuron type', ens_neurons = nengo.RectifiedLinear())
        self.param('ensemble synapse', ens_synapse = None)
        
        ## LDN Parameters 
        self.param('Set whether LDN memories are continuous', continuous=False)
        self.param('Set theta for LDN', theta=0.1)
        self.param('Set q for Reward LDN', q_r=10)
        self.param('Set q for Value LDN', q_v=10)
        self.param('Record and save ensemble spike activity', report_spikes=False)
        
        ## Extra
        self.param('generate encoders by sampling state space', sample_encoders=False)
        self.param('Set number of dimensions', dims=None)
         
    
    def evaluate(self, param):
        ## Get start time
        time1 = time.time()
        
        ## Lists for saving data
        Ep_rewards=[]
        Rewards=[]
        Values=[] 
        Roll_mean = []
        States = []
        Spikes=[]
        
        ## Set environment
        env = sparserl.wrappers.MiniGrid_wrapper(param.env, param.dist)
            
        ## Set parameters
        n_actions = env.action_space.n
        
        env_dt=param.env_dt
        trials=param.trials
        steps=param.steps
        
        dist=param.dist
        rep=param.rep
        
        rule=param.rule
        lr=param.lr
        act_discount=param.act_discount
        state_discount=param.state_discount
        n=param.n
        lambd=param.lambd
        
        n_neurons=param.n_neurons
        sparsity=param.sparsity
        ens_neurons=param.ens_neurons
        ens_synapse=param.ens_synapse
        
        continuous=param.continuous
        theta=param.theta
        q_r=param.q_r
        q_v=param.q_v
        report_spikes=param.report_spikes
        
        ## Set sample encoders to True to generate place cells
        if param.sample_encoders == True and n_neurons is not None:
            pts = np.random.uniform(0,1,size=(n_neurons,len(env.observation_space.high)))
            pts = pts * (env.observation_space.high-env.observation_space.low)
            pts = pts + env.observation_space.low
            encoders = [rep.map(rep.get_state(x, env)).copy() for x in pts]
        else:
            encoders = nengo.Default
        
        ## Choose the AC network 
        if rule == sparserl.rules.ActorCriticTDtheta:
            ac = sparserl.networks.ActorCriticLDN(rep, 
                                rule(n_actions=n_actions, alpha=alpha, beta=beta, gamma=gamma, n=n, lambd=lambd, env_dt=env_dt), 
                                n_neurons=n_neurons, sparsity=sparsity, ens_neurons=ens_neurons, ens_synapse=ens_synapse,
                                theta=theta, q_r=q_r, q_v=q_v, continuous=continuous, 
                                report_ldn=report_ldn, encoders=encoders)
        else:
            ac = sparserl.networks.ActorCritic(rep, 
                                               rule(n_actions=n_actions, alpha=lr, beta=act_discount, gamma=state_discount, 
                                                    n=n, lambd=lambd), 
                                               n_neurons=n_neurons, sparsity=sparsity, report_spikes=report_spikes, encoders=encoders)          
        
        ## LEARNING ##
        for trial in tqdm(range(trials)):
            
            if trial == trials-1 and report_spikes == True:
                report = True
            else:
                report = False

            rs=[] ##reward storage
            vs=[] ##value storage
            sts=[] ##state storage
            ims=[] ##render storage
            env.reset() ##reset environment
            update_state = rep.get_state(env.reset(), env) ##get state

            if report_spikes == True:
                value, action_logits, spikes = ac.step(update_state, 0, 0, reset=True, report=report_spikes) ##get state and action values
            else:
                value, action_logits = ac.step(update_state, 0, 0, reset=True) ##get state and action values

            ## For each time step
            for i in range(steps): 
                ## Optional command to render the environment every n trials
                #if trial % (trials/20) == 0:
                #    env.render()

                ## Choose and do action
                action_distribution = softmax(action_logits) 
                action = np.random.choice(n_actions, 1, p=action_distribution)
                obs, reward, done, info = env.step(int(action))

                ## Get new state
                current_state = rep.get_state(obs, env)

                ## Update state and action values 
                if report_spikes==True:
                    value, action_logits, spikes = ac.step(current_state, action, reward, report=report_spikes)
                else:
                    value, action_logits = ac.step(current_state, action, reward)

                rs.append(reward) ##save reward
                vs.append(value.copy()) ##save state value
                sts.append(current_state) ##save state

                ## When using mutli-step back-ups, once the agent has reached the goal you 
                ## need it to sit there for n time steps and continue doing updates
                if done:
                    break                    

            Ep_rewards.append(np.sum(rs)) ##Store average reward for episode
            Rewards.append(rs) ##Store all rewards in episode
            Values.append(vs) ##Store all values in episode  
            States.append(sts) ##Store all states in episode
        
        ## Get end time
        time2 = time.time()
        ## Get total time elapsed
        elapse_time = time2-time1
        
        ## Convert list of rewards per episode to dataframe    
        rewards_over_eps = pd.DataFrame(Ep_rewards)
        ## Calculate a rolling average reward across previous 100 episodes
        Roll_mean.append(rewards_over_eps[rewards_over_eps.columns[0]].rolling(100).mean())
        
        if report_spikes == True:
            for i in range(len(spikes)):
                sps = []
                for j in range(len(spikes[i])):
                    sps.append(spikes[i][j])
                Spikes.append([sps])
        
        return dict(
            episodes = Ep_rewards,
            rewards = Rewards,
            values = Values,
            states = States,
            spikes = Spikes,
            duration = elapse_time,
            )