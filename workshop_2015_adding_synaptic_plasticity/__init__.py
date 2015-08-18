import model_binaries


def _init_module():
    import logging
    import os
    import spynnaker.pyNN

    # Register this path with SpyNNaker
    spynnaker.pyNN.register_binary_search_path(os.path.dirname(
        model_binaries.__file__))

_init_module()

# Import rules with PyNNlike names
from spike_pair_time_dependency import SpikePairTimeDependency as SpikePairRule
from spike_triplet_time_dependency import SpikeTripletTimeDependency as SpikeTripletRule