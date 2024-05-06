from trial import ActorCriticLearn
import nni
import sparserl

def main(args):
    out = ActorCriticLearn().run(trials=500,
                                 lr=0.55, act_discount=0.87, state_discount=0.97,
                                 n_neurons = args['n_neurons'], sparsity = args['sparsity'], report_spikes=True,
                                 verbose=False, seed=args['seed'], data_dir='.\\grid_search_sparsity')
    result=[val for index,val in enumerate(out['roll_mean'][0])][-1]
    nni.report_final_result(result)
                                       
if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)