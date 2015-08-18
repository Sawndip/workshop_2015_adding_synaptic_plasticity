#include "timing_triplet_impl.h"

//---------------------------------------
// Globals
//---------------------------------------
// Exponential lookup-tables
int16_t tau_x_lut[TAU_X_LUT_SIZE];
int16_t tau_y_lut[TAU_Y_LUT_SIZE];

//---------------------------------------
// Functions
//---------------------------------------
address_t timing_initialise(address_t address)
{
    log_info("timing_initialise: starting");
    log_info("\tSTDP triplet rule");

    // Copy LUTs from memory starting at address
    address_t lut_address = maths_copy_int16_lut(&address[0],
        TAU_X_LUT_SIZE, &tau_x_lut[0]);
    lut_address = maths_copy_int16_lut(lut_address,
        TAU_Y_LUT_SIZE, &tau_y_lut[0]);

    log_info("timing_initialise: completed successfully");

    return lut_address;
}
