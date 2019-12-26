import os

from pprint import pprint
from brian2 import *
from matplotlib.pyplot import *

def generate_network(
        inputs,
        input_rate = 100*Hz,
        weight = .3,
        run_time = 200,
        part_length = 10,
        I=1,
        tau_range = linspace(1, 1, 30)*ms,
        name = "SNN Network",
        output_path = os.path.join(os.environ["OUTPUT_PATH"], "out")
):
    num_inputs = 0
    if type(inputs) is np.ndarray and len(inputs) > 0 and type(inputs[0]) is np.ndarray:
        num_inputs = len(inputs[0])
    else:
        exit(0)
    num_tau = len(tau_range)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    part_count = int(run_time / part_length)

    result = []
    sample_count = 0
    for sample in inputs:
        start_scope()
        print("Sample: ", sample)
        # We make tau a parameter of the group
        eqs = '''
        dv/dt = (I-v)/tau : 1
        I: 1
        tau : second
        '''
        # And we have num_tau output neurons, each with a different tau
        G = NeuronGroup(num_tau, eqs, threshold='v>1', reset='v=0', method='exact')
        G.tau = tau_range
        G.I = I
        SS = Synapses(G, G, on_pre='v += (weight + randn())')
        SS.connect()

        for i in range(0, num_inputs):
            P = PoissonGroup(
                1,
                rates= sample[i] * input_rate
            )
            S = Synapses(P, G, on_pre="v += .2")
            S.connect()

        M = SpikeMonitor(G)
        # Now we can just run once with no loop
        run(run_time * ms)
        output_rates = M.count/second # firing rate is count/duration


        neuron_fire_d = np.zeros(shape=(num_tau, part_count))
        index = 0
        for fire_point in M.i:
            s_time = (M.t / ms)[index]
            s_index = int(s_time / part_length)
            neuron_fire_d[fire_point][s_index] += 1
            index += 1

        sample_count += 1
        result.append({
            "sample": sample,
            "fire_rate": neuron_fire_d,
        })
    return result