from spynnaker.pyNN.models.neural_properties.synapse_dynamics.abstract_rules.\
    abstract_time_dependency import AbstractTimeDependency
from spynnaker.pyNN.models.neural_properties.synapse_dynamics.\
    plastic_weight_synapse_row_io import PlasticWeightSynapseRowIo
from spynnaker.pyNN.models.neural_properties.synapse_dynamics\
    import plasticity_helpers

import logging
logger = logging.getLogger(__name__)

# Constants
# **NOTE** these should be passed through magical per-vertex build setting
# thing
LOOKUP_TAU_PLUS_SIZE = 256
LOOKUP_TAU_PLUS_SHIFT = 0
LOOKUP_TAU_MINUS_SIZE = 256
LOOKUP_TAU_MINUS_SHIFT = 0


class SpikePairTimeDependency(AbstractTimeDependency):

    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        AbstractTimeDependency.__init__(self)

        self._tau_plus = tau_plus
        self._tau_minus = tau_minus

    def __eq__(self, other):
        if (other is None) or (not isinstance(other, SpikePairTimeDependency)):
            return False
        return ((self._tau_plus == other.tau_plus) and
                (self._tau_minus == other.tau_minus))

    def create_synapse_row_io(
            self, synaptic_row_header_words, dendritic_delay_fraction):
        return PlasticWeightSynapseRowIo(
            synaptic_row_header_words, dendritic_delay_fraction)

    def get_params_size_bytes(self):
        return 2 * (LOOKUP_TAU_PLUS_SIZE + LOOKUP_TAU_MINUS_SIZE)

    def is_time_dependance_rule_part(self):
        return True

    def write_plastic_params(self, spec, machine_time_step, weight_scales,
                             global_weight_scale):

        # Check timestep is valid
        if machine_time_step != 1000:
            raise NotImplementedError("STDP LUT generation currently only "
                                      "supports 1ms timesteps")

        # Write lookup tables
        plasticity_helpers.write_exp_lut(spec, self.tau_plus,
                                         LOOKUP_TAU_PLUS_SIZE,
                                         LOOKUP_TAU_PLUS_SHIFT)
        plasticity_helpers.write_exp_lut(spec, self.tau_minus,
                                         LOOKUP_TAU_MINUS_SIZE,
                                         LOOKUP_TAU_MINUS_SHIFT)

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

    @property
    def tau_plus(self):
        return self._tau_plus

    @property
    def tau_minus(self):
        return self._tau_minus
