#ifndef NNET_CONV_OUTPUT_REUSE_H_
#define NNET_CONV_OUTPUT_REUSE_H_

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

//output_reuse
template<typename CONFIG_T>
void conv_output_reuse1x1(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				partial_sum[oh][ow] += feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0]* weight[ic][0][0];
			}
		}
	}
};

//output_reuse
template<typename CONFIG_T>
void conv_output_reuse3x1(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				partial_sum[oh][ow] += mac3<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[ic][0][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[ic][1][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[ic][2][0]);
			}
		}
	}
};

//output_reuse
template<typename CONFIG_T>
void conv_output_reuse1x3(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				partial_sum[oh][ow] += mac3<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[ic][0][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[ic][0][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[ic][0][2]);
			}
		}
	}
};
//output_reuse
template<typename CONFIG_T>
void conv_output_reuse3x3(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
	#pragma HLS array_partition variable=feature_in  complete dim=2
	#pragma HLS array_partition variable=feature_in  complete dim=3
	#pragma HLS array_partition variable=weight      complete dim=2
	#pragma HLS array_partition variable=weight      complete dim=3
	#pragma HLS array_partition variable=partial_sum complete dim=1
	#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
				#pragma HLS pipeline
				partial_sum[oh][ow] += mac9<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[ic][0][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[ic][1][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[ic][2][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[ic][0][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1], weight[ic][1][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1], weight[ic][2][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[ic][0][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2], weight[ic][1][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2], weight[ic][2][2]);
			}
		}
	}		
};


//output_reuse
template<typename CONFIG_T>
void conv_output_reuse7x1(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				partial_sum[oh][ow] += mac7<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[ic][0][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[ic][1][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[ic][2][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 0], weight[ic][3][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 0], weight[ic][4][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 5][ow*CONFIG_T::stride_width + 0], weight[ic][5][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 6][ow*CONFIG_T::stride_width + 0], weight[ic][6][0]);
			}
		}
	}
};

//output_reuse
template<typename CONFIG_T>
void conv_output_reuse1x7(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				partial_sum[oh][ow] += mac7<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[ic][0][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[ic][0][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[ic][0][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 3], weight[ic][0][3],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 4], weight[ic][0][4],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 5], weight[ic][0][5],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 6], weight[ic][0][6]);
			}
		}
	}
};
//output_reuse
template<typename CONFIG_T>
void conv_output_reuse5x5(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				partial_sum[oh][ow] += mac25<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], weight[ic][0][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1], weight[ic][0][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2], weight[ic][0][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 3], weight[ic][0][3],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 4], weight[ic][0][4],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], weight[ic][1][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1], weight[ic][1][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2], weight[ic][1][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 3], weight[ic][1][3],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 4], weight[ic][1][4],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], weight[ic][2][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1], weight[ic][2][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2], weight[ic][2][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 3], weight[ic][2][3],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 4], weight[ic][2][4],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 0], weight[ic][3][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 1], weight[ic][3][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 2], weight[ic][3][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 3], weight[ic][3][3],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 4], weight[ic][3][4],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 0], weight[ic][4][0],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 1], weight[ic][4][1],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 2], weight[ic][4][2],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 3], weight[ic][4][3],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 4], weight[ic][4][4]);
			}
		}
	}
};


//output_reuse
template<typename CONFIG_T>
void conv_output_reuse7x7(typename CONFIG_T::in_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::weight_t weight[CONFIG_T::n_chan][CONFIG_T::filt_height][CONFIG_T::filt_width],
	typename CONFIG_T::out_t partial_sum[CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=weight      complete dim=2
#pragma HLS array_partition variable=weight      complete dim=3
#pragma HLS array_partition variable=partial_sum complete dim=1
#pragma HLS array_partition variable=partial_sum complete dim=2
	//conv2d
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			//output reuse
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
				typename CONFIG_T::accum_t row_sum[CONFIG_T::filt_height] = { 0 };
				#pragma HLS pipeline
				for (int kh = 0; kh < CONFIG_T::filt_height; kh++) {
					row_sum[kh] = mac7<CONFIG_T>(
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 0], weight[ic][kh][0],
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 1], weight[ic][kh][1],
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 2], weight[ic][kh][2],
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 3], weight[ic][kh][3],
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 4], weight[ic][kh][4],
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 5], weight[ic][kh][5],
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 6], weight[ic][kh][6]);
				}
				partial_sum[oh][ow] += sum7<CONFIG_T>(row_sum[0], row_sum[1], row_sum[2], row_sum[3], row_sum[4], row_sum[5], row_sum[6]);	
			}
		}
	}
};

} // end namespace

#endif