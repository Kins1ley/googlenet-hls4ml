#ifndef NNET_LINEAR_H_
#define NNET_LINEAR_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet{
    struct linear_config {
        typedef float in_type;
        typedef float out_type;
        typedef float weight_type;

        static const unsigned in_size;
        static const unsigned out_size;
    };

template<typename CONFIG_T>
void linear(
    typename CONFIG_T::in_type feature_in[CONFIG_T::in_size],
    typename CONFIG_T::weight_type weight_in[CONFIG_T::in_size][CONFIG_T::out_size],
    typename CONFIG_T::out_type feature_out[CONFIG_T::out_size])
    {
        for(int i_idx = 0; i_idx < CONFIG_T::in_size; ++i_idx)
        {
            for(int o_idx = 0; o_idx < CONFIG_T::out_size; ++o_idx)
            {
#pragma HLS UNROLL
                feature_out[o_idx] 
                    += (typename CONFIG_T::out_type)(feature_in[i_idx] * weight_in[i_idx][o_idx]);
            }
        }
    } // end linear

} // end namespace

#endif