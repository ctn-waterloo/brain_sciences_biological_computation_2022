import nengo
import numpy as np
try: 
    import learn_dyn_sys
except ImportError:
    raise ImportError('learn_dyn_sys is a private repository.  To request access, please contact celiasmith@uwaterloo.ca')

class Environment(object):
    x = 0
    y = 1
    vx = 2#np.random.uniform(0, 2)
    vy = 2#np.random.uniform(0, 2)

    def __init__(self, pred_steps=10):
        self.pred_steps = pred_steps

    
    def update(self, t, predict):
        dt = 0.001
        #self.vx += np.random.normal(0, 0.01)
        #self.vy += np.random.normal(0, 0.01)
        #v = np.sqrt(self.vx**2+self.vy**2)
        #self.vx /= v
        #self.vy /= v
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        while self.x > 1:
            self.x = 1 - (self.x-1)
            self.vx = -self.vx
        while self.x < -1:
            self.x = -1 - (self.x+1)
            self.vx = -self.vx
        while self.y > 1:
            self.y = 1 - (self.y-1)
            self.vy = -self.vy
        while self.y < -1:
            self.y = -1 - (self.y+1)
            self.vy = -self.vy

        path = []
        for i in range(self.pred_steps):
            path.append('<circle cx={} cy={} r=1 style="fill:yellow"/>'.format(predict[2*i]*100,predict[2*i+1]*100))

        Environment.update._nengo_html_ = '''
        <svg width=100% height=100% viewbox="-100 -100 200 200">
            <rect x=-100 y=-100 width=200 height=200 style="fill:green"/>
            <circle cx={} cy={} r=5 style="fill:white"/>
            {}           
        </svg>
        '''.format(self.x*100, self.y*100, ''.join(path))
            
        return self.x, self.y, self.vx, self.vy



def run_trial(q=10, theta=0.5, pred_steps=10, duration=10, neural=False, seed=0):

    model = nengo.Network(seed=seed)
    with model:
        env = Environment(pred_steps=pred_steps)


        c_lmu = nengo.Node(learn_dyn_sys.LMU(q=q, theta=theta, size_in=2))
    
    

        
        llp = learn_dyn_sys.LearnDynSys(size_c=2*q, size_z=2, q=q, theta=theta, 
                                        n_neurons=1000, learning_rate=1e-4,
                                        radius=2,
                                        intercepts=nengo.dists.CosineSimilarity(2*q+2))
        env_node = nengo.Node(env.update, size_in=2 * pred_steps)
    
        nengo.Connection(env_node[:2], llp.z, synapse=None)
        nengo.Connection(env_node[:2], c_lmu, synapse=None)
        if neural:
            ens = nengo.networks.EnsembleArray(20,q*2,ens_dimensions=1)
            nengo.Connection(c_lmu, ens.input, synapse=None)
            nengo.Connection(ens.output, llp.c)
        else:
            nengo.Connection(c_lmu, llp.c, synapse=None)
    
        pred_x = nengo.Node(size_in=pred_steps)
        nengo.Connection(llp.Z[:q], pred_x, transform=llp.get_weights_for_delays(np.linspace(0, 1, pred_steps)), synapse=None)
        pred_y = nengo.Node(size_in=pred_steps)
        nengo.Connection(llp.Z[q:], pred_y, transform=llp.get_weights_for_delays(np.linspace(0, 1, pred_steps)), synapse=None)
    
        nengo.Connection(pred_x, env_node[::2], synapse=0)
        nengo.Connection(pred_y, env_node[1::2], synapse=0)

        p_env = nengo.Probe(env_node)
        p_pred_x = nengo.Probe(pred_x)
        p_pred_y = nengo.Probe(pred_y)
    ### end with 

    with nengo.Simulator(model) as sim:
        sim.run(duration)
    ### end with

    return sim.trange(), sim.data[p_env], sim.data[p_pred_x], sim.data[p_pred_y]

if __name__=='__main__':
    import os.path
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser(add_help=True)

    parser.add_argument('--q', type=int, default=10, help='Dimensionality of LDN representation.')
    parser.add_argument('--pred_steps', type=int, default=10, help='Number of steps to predict')
    parser.add_argument('--theta', type=float, default=0.5, help='Prediction window (seconds)')
    parser.add_argument('--duration', type=float, default=1000, help='Simulation duration (seconds)')
    parser.add_argument('--dest_dir', type=str, default=f'../../data/isolated-llp/ball-model/', help='Location of output data')

    args = parser.parse_args()
    times, env, pred_x, pred_y = run_trial(q=args.q, theta=args.theta, pred_steps=args.pred_steps, duration=args.duration)

    np.savez(os.path.join(args.dest_dir,f'ball-llp-trial-{time.strftime("%Y%m%d-%H%M%S")}.npz'),time=times, env=env, pred_x=pred_x, pred_y=pred_y)
    
