#ifndef NNET_BIAS_H_
#define NNET_BIAS_H_

#include "nnet_common.h"
#include <cstdlib>
namespace nnet {

	struct set_bias_config {
		typedef float bias_type;
		typedef float feature_type;

		static const unsigned channel = 100;
		static const unsigned width = 5;
		static const unsigned height = 5;
	};


	template<typename CONFIG_T>
	void set_bias(
		typename CONFIG_T::feature_type buffer[CONFIG_T::channel][CONFIG_T::height][CONFIG_T::width],
		typename CONFIG_T::bias_type bias[CONFIG_T::channel])
	{

		for (int h_idx = 0; h_idx < CONFIG_T::height; ++h_idx)
		{
			for (int w_idx = 0; w_idx < CONFIG_T::width; ++w_idx)
			{
#pragma HLS PIPELINE
				for (int i_idx = 0; i_idx < CONFIG_T::channel; ++i_idx)
				{
#pragma HLS UNROLL
					buffer[i_idx][h_idx][w_idx] = (typename CONFIG_T::feature_type)(bias[i_idx]);
				}
			}
		}
	} // end set_bias
}
#endif