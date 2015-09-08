#ifndef _TIMING_TRIPLET_IMPL_H_
#define _TIMING_TRIPLET_IMPL_H_

//---------------------------------------
// Typedefines
//---------------------------------------
typedef struct post_trace_t
{
    int16_t y1;
    int16_t y2;
} post_trace_t;

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

#define TAU_Y1_LUT_SHIFT 0
#define TAU_Y1_LUT_SIZE 256

#define TAU_Y2_LUT_SHIFT 2
#define TAU_Y2_LUT_SIZE 256

// Helper macros for looking up decays
#define DECAY_TAU_X(t) \
    maths_lut_exponential_decay(t, TAU_X_LUT_SHIFT, TAU_X_LUT_SIZE, tau_x_lut)
#define DECAY_TAU_Y1(t) \
    maths_lut_exponential_decay(t, TAU_Y1_LUT_SHIFT, TAU_Y1_LUT_SIZE, tau_y1_lut)
#define DECAY_TAU_Y2(t) \
    maths_lut_exponential_decay(t, TAU_Y2_LUT_SHIFT, TAU_Y2_LUT_SIZE, tau_y2_lut)

//---------------------------------------
// Externals
//---------------------------------------
extern int16_t tau_x_lut[TAU_X_LUT_SIZE];
extern int16_t tau_y1_lut[TAU_Y1_LUT_SIZE];
extern int16_t tau_y2_lut[TAU_Y2_LUT_SIZE];

//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace()
{
    return (post_trace_t) {.y1 = 0, .y2 = 0};
}
//---------------------------------------
static inline post_trace_t timing_add_post_spike(uint32_t time,
    uint32_t last_time, post_trace_t last_trace)
{
    // Get time since last spike
    uint32_t delta_time = time - last_time;

    // Decay previous trace (y1)
    int32_t new_y1 = STDP_FIXED_MUL_16X16(last_trace.y1,
        DECAY_TAU_Y1(delta_time));

    // Add energy caused by new spike to traces
    new_y1 += STDP_FIXED_POINT_ONE;

    // Y2 is sampled in timing_apply_post_spike BEFORE the spike
    // Therefore, if this is the first spike, y2 must be zero
    int32_t new_y2;
    if(last_time == 0)
    {
        new_y2 = 0;
    }
    // Otherwise, add energy of spike to last value and decay
    else
    {
        new_y2 = STDP_FIXED_MUL_16X16(last_trace.y2 + STDP_FIXED_POINT_ONE,
            DECAY_TAU_Y2(delta_time));
    }

    log_debug("\tdelta_time=%d, y1=%d, y2=%d\n", delta_time, new_y1, new_y2);

    // Return new trace_value
    return (post_trace_t) {.y1 = new_y1, .y2 = new_y2};
}

//---------------------------------------
static inline pre_trace_t timing_add_pre_spike(uint32_t time,
    uint32_t last_time, pre_trace_t last_trace)
{
    // Get time since last spike
    uint32_t delta_time = time - last_time;

    // Decay previous x trace and add energy caused by new spike to trace
    int32_t new_x = STDP_FIXED_MUL_16X16(last_trace, DECAY_TAU_X(delta_time));
    new_x += STDP_FIXED_POINT_ONE;

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
        // Calculate y1(time) = y1(last_post_time) * e^(-delta_t/tau_y)
        int32_t y1 = STDP_FIXED_MUL_16X16(last_post_trace.y1,
            DECAY_TAU_Y1(delta_t));

        log_debug("\t\t\tdelta_t=%u, y1=%d\n", delta_t, y1);

        // Return synaptic state after applying depression
        return weight_one_term_apply_depression(previous_state, y1);
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

        // Multiply this by y2(time) to get triplet term
        int32_t triplet = STDP_FIXED_MUL_16X16(x, trace.y2);

        log_debug("\t\t\tdelta_t=%u, x=%d, y2=%d, triplet=%d\n",
                  delta_t, x, trace.y2, triplet);

        // Apply potentiation to synapse state
        return weight_one_term_apply_potentiation(previous_state, triplet);
    }
    // Otherwise, return unmodified synaptic state
    else
    {
        return previous_state;
    }
}

#endif // _TIMING_TRIPLET_IMPL_H_
