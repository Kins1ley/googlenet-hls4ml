#ifndef NNET_LRN_H_
#define NNET_LRN_H_

#include "nnet_common.h"
#include <cstdlib>
#include <cmath>

namespace nnet {
    struct lrn_config {

        typedef float in_type;
        typedef float out_type;

        static const unsigned in_channel    = 10;
        static const unsigned in_height     = 10;
        static const unsigned in_width      = 10;
    };

template<typename CONFIG_T>
void LRN(
    typename CONFIG_T::in_type feature_in[CONFIG_T::in_channel][CONFIG_T::in_height][CONFIG_T::in_width],
    typename CONFIG_T::out_type feature_out[CONFIG_T::in_channel][CONFIG_T::in_height][CONFIG_T::in_width], 
    float bias, float alpha, float beta, int N)
    {
        for (int i_idx = 0; i_idx < CONFIG_T::in_channel; i_idx++)
        {
            int lower_bound = (i_idx - N) > 0 ? (i_idx - N) : 0;
            int upper_bound = (i_idx + N) > CONFIG_T::in_channel ? CONFIG_T::in_channel : (i_idx + N);
            for (int h_idx = 0; h_idx < CONFIG_T::in_height; h_idx++)
            {
                for (int w_idx = 0; w_idx < CONFIG_T::in_width; w_idx++)
                {
                    typename CONFIG_T::in_type summary = (typename CONFIG_T::in_type)0;
                    typename CONFIG_T::in_type temp[N*2];
                    for (int c_idx = lower_bound; c_idx < upper_bound; c_idx++)
                    {
#pragma HLS PIPELINE
                        temp[c_idx] = pow(feature_in[c_idx][h_idx][w_idx], 2);
                    }
                    for (int c_idx = lower_bound; c_idx < upper_bound; c_idx++)
                    {
#pragma HLS PIPELINE
                        summary += temp[c_idx];
                    }
                    feature_out[i_idx][h_idx][w_idx] 
                        = (typename CONFIG_T::out_type)(feature_in[i_idx][h_idx][w_idx] / pow(bias + alpha * summary, beta)) ;
                }
            }   
        }
    } // end LRN

} // end namespace

#endif