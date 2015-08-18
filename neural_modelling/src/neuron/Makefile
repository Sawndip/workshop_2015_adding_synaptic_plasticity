MODELS = IF_curr_exp_stdp_mad_pair_additive IF_curr_exp_stdp_mad_triplet_additive
BUILD_DIRS := $(addprefix builds/, $(MODELS))

all: $(BUILD_DIRS)
	for d in $(BUILD_DIRS); do (cd $$d; "$(MAKE)") || exit $$?; done

clean: $(BUILD_DIRS)
	for d in $(BUILD_DIRS); do (cd $$d; "$(MAKE)" clean) || exit $$?; done
