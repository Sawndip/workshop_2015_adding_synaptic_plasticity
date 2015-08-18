from spynnaker.pyNN.models.neural_properties.synapse_dynamics.abstract_rules.\
    abstract_time_dependency import AbstractTimeDependency
from spynnaker.pyNN.models.neural_properties.synapse_dynamics.\
    plastic_weight_synapse_row_io import PlasticWeightSynapseRowIo
from spynnaker.pyNN.models.neural_properties.synapse_dynamics\
    import plasticity_helpers

import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TAU_X_LUT_SHIFT = 0
TAU_X_LUT_SIZE = 256

TAU_Y_LUT_SHIFT = 0
TAU_Y_LUT_SIZE = 256

# ----------------------------------------------------------------------------
# SpikeTripletTimeDependency
# ----------------------------------------------------------------------------
class SpikeTripletTimeDependency(AbstractTimeDependency):
    def __init__(self, tau_x=20.0, tau_y=20.0):
        AbstractTimeDependency.__init__(self)

        self.tau_x = tau_x
        self.tau_y = tau_y

    def __eq__(self, other):
        if (other is None) or (not isinstance(other, TripletTimeDependency)):
            return False
        return ((self.tau_x == other.tau_x) and
                (self.tau_y == other.tau_y))

    def create_synapse_row_io(self, synaptic_row_header_words,
                              dendritic_delay_fraction):
        return PlasticWeightSynapseRowIo(
            synaptic_row_header_words, dendritic_delay_fraction)

    def get_params_size_bytes(self):
        # 2 bytes for each entry in each LUT
        return 2 * (TAU_X_LUT_SIZE + TAU_Y_LUT_SIZE)

    def is_time_dependance_rule_part(self):
        return True

    def write_plastic_params(self, spec, machine_time_step, weight_scales,
                             global_weight_scale):

        # Check timestep is valid
        if machine_time_step != 1000:
            raise NotImplementedError("STDP LUT generation currently only "
                                      "supports 1ms timesteps")

        # Write lookup tables
        plasticity_helpers.write_exp_lut(spec, self.tau_x,
                                         TAU_X_LUT_SIZE,
                                         TAU_X_LUT_SHIFT)
        plasticity_helpers.write_exp_lut(spec, self.tau_y,
                                         TAU_Y_LUT_SIZE,
                                         TAU_Y_LUT_SHIFT)

    @property
    def num_terms(self):
        return 1

    @property
    def vertex_executable_suffix(self):
        return "pair"

    @property
    def pre_trace_size_bytes(self):
        # A single 16-bit pre-synaptic trace is required
        return 2
