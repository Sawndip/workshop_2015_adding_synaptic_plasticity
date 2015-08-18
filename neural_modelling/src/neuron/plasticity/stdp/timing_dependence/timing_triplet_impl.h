#ifndef _TIMING_TRIPLET_IMPL_H_
#define _TIMING_TRIPLET_IMPL_H_

//---------------------------------------
// Typedefines
//---------------------------------------
typedef int16_t post_trace_t;
typedef int16_t pre_trace_t;


#include "neuron/plasticity/stdp/timing_dependence/timing.h"
#include "neuron/plasticity/stdp/weight_dependence/weight_one_term.h"

// Include debug header for log_info etc
#include <debug.h>

// Include generic plasticity maths functions
#include "neuron/plasticity/common/maths.h"
#include "neuron/plasticity/common/stdp_typedefs.h"

//---------------------------------------
// Macros
//---------------------------------------
// Exponential decay lookup parameters
#define TAU_X_LUT_SHIFT 0
#define TAU_X_LUT_SIZE 256

#define TAU_Y_LUT_SHIFT 0
#define TAU_Y_LUT_SIZE 256

// Helper macros for looking up decays
#define DECAY_TAU_X(t) \
    maths_lut_exponential_decay(t, TAU_X_LUT_SHIFT, TAU_X_LUT_SIZE, tau_x_lut)
#define DECAY_TAU_Y(t) \
    maths_lut_exponential_decay(t, TAU_Y_LUT_SHIFT, TAU_Y_LUT_SIZE, tau_y_lut)

//---------------------------------------
// Externals
//---------------------------------------
extern int16_t tau_x_lut[TAU_X_LUT_SIZE];
extern int16_t tau_y_lut[TAU_Y_LUT_SIZE];

//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace()
{
    return 0;
}
//---------------------------------------
static inline post_trace_t timing_add_post_spike(uint32_t time,
    uint32_t last_time, post_trace_t last_trace)
{
    // Get time since last spike
    uint32_t delta_time = time - last_time;

    // Decay previous trace (y)
    int32_t new_y = STDP_FIXED_MUL_16X16(last_trace, DECAY_TAU_Y(delta_time));

    // Add energy caused by new spike to trace
    new_y += STDP_FIXED_POINT_ONE;

    log_debug("\tdelta_time=%d, y=%d\n", delta_time, new_y);

    // Return new trace_value
    return (post_trace_t)new_y;
}

//---------------------------------------
static inline pre_trace_t timing_add_pre_spike(uint32_t time,
    uint32_t last_time, pre_trace_t last_trace, bool flush)
{
    // Get time since last spike
    uint32_t delta_time = time - last_time;

    // Decay previous x trace
    int32_t new_x = STDP_FIXED_MUL_16X16(last_trace, DECAY_TAU_X(delta_time));

    // If this isn't a flush, add energy caused by new spike to trace
    if(!flush)
    {
        new_x += STDP_FIXED_POINT_ONE;
    }

    log_debug("\tdelta_time=%u, x=%d\n", delta_time, new_x);

    // Return new trace_value
    return (pre_trace_t)new_x;
}

//---------------------------------------
static inline update_state_t timing_apply_pre_spike(uint32_t time,
    pre_trace_t trace, uint32_t last_pre_time, pre_trace_t last_pre_trace,
    uint32_t last_post_time, post_trace_t last_post_trace,
    update_state_t previous_state)
{
    use(&trace);
    use(last_pre_time);
    use(&last_pre_trace);

    // Get time of pre-synaptic spike relative
    // to time of last post-synaptic spike
    uint32_t delta_t = time - last_post_time;

    // If spikes are not co-incident
    if (delta_t > 0)
    {
        // Calculate y(time) = y(last_post_time) * e^(-delta_t/tau_y)
        int32_t y = STDP_FIXED_MUL_16X16(last_post_trace,
            DECAY_TAU_Y(delta_t));

        log_debug("\t\t\tdelta_t=%u, y=%d\n", delta_t, y);

        // Return synaptic state after applying depression
        return weight_one_term_apply_depression(previous_state, y);
    }
    // Otherwise, return unmodified synaptic state
    else
    {
        return previous_state;
    }
}

//---------------------------------------
static inline update_state_t timing_apply_post_spike(
    uint32_t time, post_trace_t trace, uint32_t last_pre_time,
    pre_trace_t last_pre_trace, uint32_t last_post_time,
    post_trace_t last_post_trace, update_state_t previous_state)
{
    use(&trace);
    use(last_post_time);
    use(&last_post_trace);

    // Get time of post-synaptic spike relative
    // to time of last pre-synaptic spike
    uint32_t delta_t = time - last_pre_time;

    // If spikes are not co-incident
    if (delta_t > 0)
    {
        // Calculate x(time) = x(last_pre_time) * e^(-delta_t/tau_x)
        int32_t x = STDP_FIXED_MUL_16X16(last_pre_trace,
            DECAY_TAU_X(delta_t));

        log_debug("\t\t\tdelta_t=%u, x=%d\n",
                  delta_t, x);

        // Apply potentiation to synapse state
        return weight_one_term_apply_potentiation(previous_state, x);
    }
    // Otherwise, return unmodified synaptic state
    else
    {
        return previous_state;
    }
}

#endif // _TIMING_TRIPLET_IMPL_H_
