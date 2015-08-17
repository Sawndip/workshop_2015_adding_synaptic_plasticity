APP = $(notdir $(CURDIR))
BUILD_DIR = build/
MODEL_OBJS = $(SOURCE_DIRS)/neuron/models/neuron_model_lif_curr_impl.o \
             $(SOURCE_DIRS)/neuron/plasticity/common/post_events.o \
             $(SOURCE_DIRS)/neuron/plasticity/stdp/synapse_dynamics_stdp_mad_impl.o \
             $(SOURCE_DIRS)/neuron/plasticity/common/maths.o \
             $(EXTRA_SRC_DIR)/neuron/plasticity/stdp/timing_dependence/timing_pair_impl.o \
             $(SOURCE_DIRS)/neuron/plasticity/stdp/weight_dependence/weight_additive_one_term_impl.o
NEURON_MODEL_H = $(SOURCE_DIRS)/neuron/models/neuron_model_lif_curr_impl.h
SYNAPSE_TYPE_H = $(SOURCE_DIRS)/neuron/synapse_types/synapse_types_exponential_impl.h
TIMING_DEPENDENCE_H = $(EXTRA_SRC_DIR)/neuron/plasticity/stdp/timing_dependence/timing_pair_impl.h
WEIGHT_DEPENDENCE_H = $(SOURCE_DIRS)/neuron/plasticity/stdp/weight_dependence/weight_additive_one_term_impl.h
PLASTIC_SYNAPSE_STRUCTURE_H = $(SOURCE_DIRS)/neuron/plasticity/stdp/synapse_structure/synapse_structure_weight_impl.h

include ../Makefile.common