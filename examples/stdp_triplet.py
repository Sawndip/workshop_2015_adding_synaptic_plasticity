import itertools, math, pylab
import spynnaker.pyNN as sim
import workshop_2015_adding_synaptic_plasticity as workshop

#-------------------------------------------------------------------
# This example uses the sPyNNaker implementation of the triplet rule
# Developed by Pfister and Gerstner(2006) to reproduce the pairing
# Experiment first performed by Sjostrom (2001)
#-------------------------------------------------------------------

#-------------------------------------------------------------------
# Common parameters
#-------------------------------------------------------------------
start_time = 100
time_between_pairs = 1000
num_pairs = 60

max_weight = 1.0

frequencies = [10, 20, 40, 50]
delta_t = [-10, 10]

#-------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------
def calculate_error(data, data_error, model):
    # Sum error
    error = 0.0
    for d, e, m in zip(data, data_error, model):
        error += (((d - m) / e) ** 2)
    return error / float(len(data))

def generate_fixed_frequency_test_data(frequency, first_spike_time, num_spikes):
    # Calculate interspike delays in ms
    interspike_delay = int(1000.0 / float(frequency));

    # Generate spikes
    return [first_spike_time + (s * interspike_delay) for s in range(num_spikes)]

#-------------------------------------------------------------------
# Experiment loop
#-------------------------------------------------------------------
# Population parameters
model = sim.IF_curr_exp
cell_params = {'cm'     : 0.25, # nF
            'i_offset'  : 0.0,
            'tau_m'     : 10.0,
            'tau_refrac': 2.0,
            'tau_syn_E' : 2.5,
            'tau_syn_I' : 2.5,
            'v_reset'   : -70.0,
            'v_rest'    : -65.0,
            'v_thresh'  : -55.4
            }

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# Start all connections at half maximum weight
start_w = max_weight * 0.5

# Create two STDP models, one using triplet model and one, pair
triplet_stdp_model = sim.STDPMechanism(
    timing_dependence = workshop.SpikeTripletRule(tau_x=16.8, tau_y1=33.7, tau_y2=114.0),
    weight_dependence = sim.AdditiveWeightDependence(w_min=0.0, w_max=max_weight, A_plus=6.5E-3 * start_w, A_minus=7.1E-3 * start_w)
)

pair_stdp_model = sim.STDPMechanism(
    timing_dependence = workshop.SpikePairRule(tau_x=16.8, tau_y=33.7),
    weight_dependence = sim.AdditiveWeightDependence(w_min=0.0, w_max=max_weight, A_plus=4.3E-3, A_minus=2.9E-4)
)

# Sweep times and frequencies
pair_projections = []
triplet_projections = []
sim_time = 0
for t in delta_t:
    pair_projections.append([])
    triplet_projections.append([])
    for f in frequencies:
        # Neuron populations
        pre_pop = sim.Population(1, model, cell_params)
        post_pair_pop = sim.Population(1, model, cell_params)
        post_triplet_pop = sim.Population(1, model, cell_params)

        # Stimulating populations
        pre_times = generate_fixed_frequency_test_data(f, start_time - 1, num_pairs + 1)
        post_times = generate_fixed_frequency_test_data(f, start_time + t, num_pairs)
        pre_stim = sim.Population(1, sim.SpikeSourceArray, {'spike_times': [pre_times,]})
        post_stim = sim.Population(1, sim.SpikeSourceArray, {'spike_times': [post_times,]})

        # Update simulation time
        sim_time = max(sim_time, max(max(pre_times), max(post_times)) + 100)

        # Connections between spike sources and neuron populations
        ee_connector = sim.OneToOneConnector(weights=2)
        sim.Projection(pre_stim, pre_pop, ee_connector, target='excitatory')
        sim.Projection(post_stim, post_pair_pop, ee_connector, target='excitatory')
        sim.Projection(post_stim, post_triplet_pop, ee_connector, target='excitatory')

        # Connect pre-population to both post-populations using different STDP models
        triplet_projections[-1].append(
            sim.Projection(pre_pop, post_triplet_pop,
                           sim.OneToOneConnector(weights=start_w),
                           synapse_dynamics=sim.SynapseDynamics(slow=triplet_stdp_model))
        )

        pair_projections[-1].append(
            sim.Projection(pre_pop, post_pair_pop,
                            sim.OneToOneConnector(weights=start_w),
                            synapse_dynamics=sim.SynapseDynamics(slow=pair_stdp_model))
        )

print("Simulating for %us" % (sim_time / 1000))

# Run simulation
sim.run(sim_time)

# Read weights from each parameter value being tested
pair_weights = []
triplet_weights = []
for projection_delta_t in pair_projections:
    pair_weights.append([p.getWeights()[0] for p in projection_delta_t])
for projection_delta_t in triplet_projections:
    triplet_weights.append([p.getWeights()[0] for p in projection_delta_t])

# End simulation on SpiNNaker
sim.end(stop_on_board=True)

#-------------------------------------------------------------------
# Plotting
#-------------------------------------------------------------------
# Sjostrom et al. (2001) experimental data
data_w = [
    [ -0.41, -0.34, 0.56, 0.75 ],
    [ 0.14, 0.29, 0.53, 0.56 ]
]
data_e = [
    [ 0.11, 0.1, 0.32, 0.19 ],
    [ 0.1, 0.14, 0.11, 0.26 ]
]

# Plot Frequency response
figure, axis = pylab.subplots()
axis.set_xlim((frequencies[0] - 1, frequencies[-1] + 1))
axis.set_xlabel("Frequency [Hz]")
axis.set_ylabel(r"$\frac{\Delta w_{ij}}{w_{ij}}$", rotation="horizontal", size="xx-large")
axis.axhline(color="grey", linestyle=":")

# Calculate weight deltas
pair_delta_w = [[(w - start_w) / start_w for w in m_w] for m_w in pair_weights]
triplet_delta_w = [[(w - start_w) / start_w for w in m_w] for m_w in triplet_weights]

line_styles = ["--", "-"]
for p_w, t_w, d_w, d_e, l, t in zip(pair_delta_w, triplet_delta_w, data_w, data_e, line_styles, delta_t):
    # Plot experimental data and error bars
    axis.errorbar(frequencies, d_w, yerr=d_e, color="grey", linestyle=l, label=r"Experimental data, delta $(\Delta{t}=%dms)$" % t)

    # Plot model data
    axis.plot(frequencies, p_w, color="blue", linestyle=l, label=r"Pair rule, delta $(\Delta{t}=%dms)$" % t)
    axis.plot(frequencies, t_w, color="red", linestyle=l, label=r"Triplet rule, delta $(\Delta{t}=%dms)$" % t)

axis.legend(loc="lower right")

# Calculate error
pair_error = calculate_error(data_w[0] + data_w[1],
                             data_e[0] + data_e[1],
                             itertools.chain.from_iterable(pair_delta_w))
triplet_error = calculate_error(data_w[0] + data_w[1],
                             data_e[0] + data_e[1],
                             itertools.chain.from_iterable(triplet_delta_w))

print("Pair error %f s.d., Triplet rule %f s.d." % (pair_error, triplet_error))

pylab.show()