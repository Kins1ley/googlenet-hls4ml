#ifndef NNET_CONV_INPUT_REUSE_H_
#define NNET_CONV_INPUT_REUSE_H_

#include <cstdlib>
#include "nnet_mac.h"
#include "nnet_common.h"

namespace nnet {

struct conv2d_config
{
	// Internal data type definitions
	typedef float accum_t;
	typedef float weight_t;
	typedef float in_t;
	typedef float out_t;

	// Convolutional parameters
	static const unsigned in_height = 10;
	static const unsigned in_width = 10;
	static const unsigned n_chan = 1;
	static const unsigned filt_height = 1;
	static const unsigned filt_width = 1;
	static const unsigned n_filt = 1;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = 10;
	static const unsigned out_width = 10;


};

//input_reuse
template<typename CONFIG_T>
void conv_input_reuse1x1(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
	#pragma HLS array_partition variable=feature_in  dim=1
	#pragma HLS array_partition variable=feature_in  dim=2
	#pragma HLS array_partition variable=weight      dim=1
	#pragma HLS array_partition variable=weight      dim=2
	#pragma HLS array_partition variable=weight      dim=3
	#pragma HLS array_partition variable=partial_sum dim=1
	#pragma HLS array_partition variable=partial_sum dim=2
	#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
				#pragma HLS unroll
				partial_sum[oc][oh][ow] += feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0] *weight[oc][0][0];
			}
		}
	}
};


//input_reuse
template<typename CONFIG_T>
void conv_input_reuse1x3(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
#pragma HLS unroll
				partial_sum[oc][oh][ow] += mac3<CONFIG_T>(
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[oc][0][0],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[oc][0][1],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[oc][0][2]);
			}
		}
	}
};
//input_reuse
template<typename CONFIG_T>
void conv_input_reuse3x1(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
#pragma HLS unroll
				partial_sum[oc][oh][ow] += mac3<CONFIG_T>(
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[oc][0][0],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[oc][1][0],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[oc][2][0]);
			}
		}
	}
};

//input_reuse
template<typename CONFIG_T>
void conv_input_reuse3x3(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
#pragma HLS unroll
				partial_sum[oc][oh][ow] += mac9<CONFIG_T>(
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[oc][0][0],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[oc][1][0],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[oc][2][0],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[oc][0][1],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1], weight[oc][1][1],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1], weight[oc][2][1],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[oc][0][2],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2], weight[oc][1][2],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2], weight[oc][2][2]);
			}
		}
	}
};

//input_reuse
template<typename CONFIG_T>
void conv_input_reuse1x7(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
#pragma HLS unroll
				partial_sum[oc][oh][ow] += mac7<CONFIG_T>(
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[oc][0][0],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[oc][0][1],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[oc][0][2],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 3], weight[oc][0][3],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 4], weight[oc][0][4],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 5], weight[oc][0][5],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 6], weight[oc][0][6]);
			}
		}
	}
};

//input_reuse
template<typename CONFIG_T>
void conv_input_reuse7x1(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
#pragma HLS unroll
				partial_sum[oc][oh][ow] += mac7<CONFIG_T>(
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[oc][0][0],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[oc][1][0],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[oc][2][0],
					feature_in[oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 0], weight[oc][3][0],
					feature_in[oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 0], weight[oc][4][0],
					feature_in[oh*CONFIG_T::stride_height + 5][ow*CONFIG_T::stride_width + 0], weight[oc][5][0],
					feature_in[oh*CONFIG_T::stride_height + 6][ow*CONFIG_T::stride_width + 0], weight[oc][6][0]);
			}
		}
	}
};


//input_reuse
template<typename CONFIG_T>
void conv_input_reuse5x5(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
#pragma HLS unroll
				partial_sum[oc][oh][ow] += mac25<CONFIG_T>(
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[oc][0][0],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[oc][0][1],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[oc][0][2],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 3], weight[oc][0][3],
					feature_in[oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 4], weight[oc][0][4],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[oc][1][0],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1], weight[oc][1][1],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2], weight[oc][1][2],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 3], weight[oc][1][3],
					feature_in[oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 4], weight[oc][1][4],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[oc][2][0],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1], weight[oc][2][1],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2], weight[oc][2][2],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 3], weight[oc][2][3],
					feature_in[oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 4], weight[oc][2][4],
					feature_in[oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 0], weight[oc][3][0],
					feature_in[oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 1], weight[oc][3][1],
					feature_in[oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 2], weight[oc][3][2],
					feature_in[oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 3], weight[oc][3][3],
					feature_in[oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 4], weight[oc][3][4],
					feature_in[oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 0], weight[oc][4][0],
					feature_in[oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 1], weight[oc][4][1],
					feature_in[oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 2], weight[oc][4][2],
					feature_in[oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 3], weight[oc][4][3],
					feature_in[oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 4], weight[oc][4][4] );
			}
		}
	}
};


//input_reuse
template<typename CONFIG_T>
void conv_input_reuse7x7(typename CONFIG_T::in_t feature_in[CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_filt][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::n_filt][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  dim=1
#pragma HLS array_partition variable=feature_in  dim=2
#pragma HLS array_partition variable=weight      dim=1
#pragma HLS array_partition variable=weight      dim=2
#pragma HLS array_partition variable=weight      dim=3
#pragma HLS array_partition variable=partial_sum dim=1
#pragma HLS array_partition variable=partial_sum dim=2
#pragma HLS array_partition variable=partial_sum dim=3
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
#pragma HLS pipeline
			//input reuse
			for (int oc = 0; oc < CONFIG_T::n_filt; oc++) {
			typename CONFIG_T::accum_t row_sum[CONFIG_T::filt_height] = { 0 };
			#pragma HLS unroll
				for (int kh = 0; kh < CONFIG_T::filt_height; kh++) {
					row_sum[kh] = mac7<CONFIG_T>(
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 0], weight[oc][kh][0],
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 1], weight[oc][kh][1],
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 2], weight[oc][kh][2],
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 3], weight[oc][kh][3],
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 4], weight[oc][kh][4],
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 5], weight[oc][kh][5],
						feature_in[oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 6], weight[oc][kh][6]);
				}
				partial_sum[oc][oh][ow] += sum7<CONFIG_T>(row_sum[0], row_sum[1], row_sum[2], row_sum[3], row_sum[4], row_sum[5], row_sum[6]);
			}
		}
	}
};
} // end namespace

#endif