#ifndef NNET_RELU_H_
#define NNET_RELU_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet{

    struct relu_config{

        typedef int feature_type;

        static const unsigned in_channel    = 10;
        static const unsigned in_width      = 3;
        static const unsigned in_height     = 3;
    };

template<typename CONFIG_T>
void relu_inplace(
    typename CONFIG_T::feature_type feature_in[CONFIG_T::in_channel][CONFIG_T::in_height][CONFIG_T::in_width])
    {
        for (int c_idx = 0; c_idx < CONFIG_T::in_channel; c_idx++)
        {
#pragma HLS PIPELINE
            for (int h_idx = 0; h_idx < CONFIG_T::in_height; h_idx++)
            {
#pragma HLS UNROLL
                for (int w_idx = 0; w_idx < CONFIG_T::in_width; w_idx++)
                {
#pragma HLS UNROLL
                    feature_in[c_idx][h_idx][w_idx] 
                        = (typename CONFIG_T::feature_type)(feature_in[c_idx][h_idx][w_idx] > 0 ? feature_in[c_idx][h_idx][w_idx] : 0);
                }
            }
        }
    } // end relu_inplace

template<typename CONFIG_T>
void relu(
    typename CONFIG_T::feature_type feature_in[CONFIG_T::in_channel][CONFIG_T::in_height][CONFIG_T::in_width],
    typename CONFIG_T::feature_type feature_out[CONFIG_T::in_channel][CONFIG_T::in_height][CONFIG_T::in_width])
    {
        for (int c_idx = 0; c_idx < CONFIG_T::in_channel; c_idx++)
        {
#pragma HLS PIPELINE
            for (int h_idx = 0; h_idx < CONFIG_T::in_height; h_idx++)
            {
#pragma HLS UNROLL
                for (int w_idx = 0; w_idx < CONFIG_T::in_width; w_idx++)
                {
#pragma HLS UNROLL
                    feature_out[c_idx][h_idx][w_idx] 
                        = (typename CONFIG_T::feature_type)(feature_in[c_idx][h_idx][w_idx] > 0 ? feature_in[c_idx][h_idx][w_idx] : 0);
                }
            }
        }
    } // end relu
} // end namespace

#endif