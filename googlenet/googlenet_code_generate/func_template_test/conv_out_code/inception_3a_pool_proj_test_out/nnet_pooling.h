#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

struct pool2d_config
{
	// Internal data type definitions
	typedef float feature_t;
	typedef float accum_t;

	// Convolutional parameters
	static const unsigned in_height = 10;
	static const unsigned in_width = 10;
	static const unsigned n_chan = 1;
	static const unsigned filt_height = 1;
	static const unsigned filt_width = 1;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = 10;
	static const unsigned out_width = 10;
	static const unsigned avgpool = 1;

};

template<typename CONFIG_T>
typename CONFIG_T::accum_t  average7(
	typename  CONFIG_T::feature_t  b0,
	typename CONFIG_T::feature_t  b1,
	typename CONFIG_T::feature_t  b2,
	typename CONFIG_T::feature_t  b3,
	typename CONFIG_T::feature_t  b4,
	typename CONFIG_T::feature_t  b5,
	typename CONFIG_T::feature_t  b6)
{

	typename CONFIG_T::accum_t  add00, add01, add02;
	typename CONFIG_T::accum_t  add10, add11;
	typename CONFIG_T::accum_t  add2;


	add00 = b0 + b1;
	add01 = b2 + b3;
	add02 = b4 + b5;

	add10 = add00 + add01;
	add11 = add02 + b6;

	add2 = add10 + add11;

	return add2 / (typename CONFIG_T::accum_t)7;

}
template<typename CONFIG_T>
typename CONFIG_T::feature_t  max9(
	typename  CONFIG_T::feature_t  b0,
	typename CONFIG_T::feature_t  b1,
	typename CONFIG_T::feature_t  b2,
	typename CONFIG_T::feature_t  b3,
	typename CONFIG_T::feature_t  b4,
	typename CONFIG_T::feature_t  b5,
	typename CONFIG_T::feature_t  b6,
	typename CONFIG_T::feature_t  b7,
	typename CONFIG_T::feature_t  b8)
{

	typename CONFIG_T::feature_t  max00, max01, max02,max03;
	typename CONFIG_T::feature_t  max10, max11;
	typename CONFIG_T::feature_t  max2;
	typename CONFIG_T::feature_t  max3;


	max00 = b0 > b1?b0:b1;
	max01 = b2 > b3? b2 : b3;
	max02 = b4 > b5? b4 : b5;
	max03 = b6 > b7? b6 : b7;

	max10 = max00 > max01? max00 : max01;
	max11 = max02 > max03? max02 : max03;

	max2 = max10 > max11? max10 : max11;
	max3 = max2 > b8? max2 : b8;

	return max3;

}

template<typename CONFIG_T>
typename CONFIG_T::feature_t  average9(
	typename  CONFIG_T::feature_t  b0,
	typename CONFIG_T::feature_t  b1,
	typename CONFIG_T::feature_t  b2,
	typename CONFIG_T::feature_t  b3,
	typename CONFIG_T::feature_t  b4,
	typename CONFIG_T::feature_t  b5,
	typename CONFIG_T::feature_t  b6,
	typename CONFIG_T::feature_t  b7,
	typename CONFIG_T::feature_t  b8)
{

	typename CONFIG_T::accum_t  add00, add01, add02, add03;
	typename CONFIG_T::accum_t  add10, add11;
	typename CONFIG_T::accum_t  add2;
	typename CONFIG_T::accum_t  add3;


	add00 = b0 + b1;
	add01 = b2 + b3;
	add02 = b4 + b5;
	add03 = b6 + b7;

	add10 = add00 + add01;
	add11 = add02 + add03;

	add2 = add10 + add11;
	add3 = add2 + b8;

	return add3 / (typename CONFIG_T::accum_t)9;

}

template<typename CONFIG_T>
typename CONFIG_T::feature_t  average25(
	typename  CONFIG_T::feature_t  b0,
	typename  CONFIG_T::feature_t  b1, 
	typename  CONFIG_T::feature_t  b2, 
	typename  CONFIG_T::feature_t  b3, 
	typename  CONFIG_T::feature_t  b4,
	typename  CONFIG_T::feature_t  b5,
	typename  CONFIG_T::feature_t  b6, 
	typename  CONFIG_T::feature_t  b7, 
	typename  CONFIG_T::feature_t  b8, 
	typename  CONFIG_T::feature_t  b9, 
	typename  CONFIG_T::feature_t  b10,
	typename  CONFIG_T::feature_t  b11, 
	typename  CONFIG_T::feature_t  b12, 
	typename  CONFIG_T::feature_t  b13, 
	typename  CONFIG_T::feature_t  b14,
	typename  CONFIG_T::feature_t  b15,
	typename  CONFIG_T::feature_t  b16, 
	typename  CONFIG_T::feature_t  b17, 
	typename  CONFIG_T::feature_t  b18,
	typename  CONFIG_T::feature_t  b19, 
	typename  CONFIG_T::feature_t  b20, 
	typename  CONFIG_T::feature_t  b21, 
	typename  CONFIG_T::feature_t  b22, 
	typename  CONFIG_T::feature_t  b23, 
	typename  CONFIG_T::feature_t  b24 )
{


	typename CONFIG_T::accum_t  add00, add01, add02, add03, add04, add05, add06, add07, add08, add09, add010, add011, add012, add013;
	typename CONFIG_T::accum_t  add10, add11, add12, add13, add14, add15, add16, add17, add18, add19, add110, add111, add112;
	typename CONFIG_T::accum_t  add20, add21, add22, add23, add24;
	typename CONFIG_T::accum_t  add30, add31;
	typename CONFIG_T::accum_t  add4;

	add00 = b0 + b1;
	add01 = b2 + b3;
	add02 = b4 + b5;
	add03 = b6 + b7;
	add04 = b8 + b9;
	add05 = b10 + b11;
	add06 = b12 + b13;
	add07 = b14 + b15;
	add08 = b16 + b17;
	add09 = b18 + b19;
	add010 = b20 + b21;
	add011 = b22 + b23;

	add10 = add00 + add01;
	add11 = add02 + add03;
	add12 = add04 + add05;
	add13 = add06 + add07;
	add14 = add08 + add09;
	add15 = add010 + add011;

	add20 = add10 + add11;
	add21 = add12 + add13;
	add22 = add14 + add15;

	add30 = add20 + add21;
	add31 = add22 + b24;

	add4 = add30 + add31;
	return add4 / (typename CONFIG_T::accum_t)25;

}

template<typename CONFIG_T>
void pool3x3(typename CONFIG_T::feature_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::feature_t feature_out[CONFIG_T::n_chan][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=feature_out complete dim=2
#pragma HLS array_partition variable=feature_out complete dim=3
	//avgpool
	if (CONFIG_T::avgpool){
		for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
			for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
				for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
					#pragma HLS pipeline
					feature_out[ic][oh][ow] = average9<CONFIG_T>(
						feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0], 
						feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0], 
						feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0], 
						feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1],
						feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1],
						feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1],
						feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2],
						feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2],
						feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2]);
				}
			}
		}
	}
	else {//maxpool
		for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
			for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
				for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
					feature_out[ic][oh][ow] = max9<CONFIG_T>(
						feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0],
						feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0],
						feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0],
						feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1],
						feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1],
						feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1],
						feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2],
						feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2],
						feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2]);
				}
			}
		}
	}
};

template<typename CONFIG_T>
void pool5x5(typename CONFIG_T::feature_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::feature_t feature_out[CONFIG_T::n_chan][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=feature_out complete dim=2
#pragma HLS array_partition variable=feature_out complete dim=3
	//pool
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				feature_out[ic][oh][ow] = average25<CONFIG_T>(
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 0],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 1],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 2],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 3],
					feature_in[ic][oh*CONFIG_T::stride_height + 0][ow*CONFIG_T::stride_width + 4],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 0],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 1],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 2],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 3],
					feature_in[ic][oh*CONFIG_T::stride_height + 1][ow*CONFIG_T::stride_width + 4],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 0],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 1],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 2],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 3],
					feature_in[ic][oh*CONFIG_T::stride_height + 2][ow*CONFIG_T::stride_width + 4],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 0],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 1],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 2],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 3],
					feature_in[ic][oh*CONFIG_T::stride_height + 3][ow*CONFIG_T::stride_width + 4],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 0],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 1],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 2],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 3],
					feature_in[ic][oh*CONFIG_T::stride_height + 4][ow*CONFIG_T::stride_width + 4]);
			}
		}
	}
};

template<typename CONFIG_T>
void pool7x7(typename CONFIG_T::feature_t feature_in[CONFIG_T::n_chan][CONFIG_T::in_height][CONFIG_T::in_width],
	typename CONFIG_T::feature_t feature_out[CONFIG_T::n_chan][CONFIG_T::out_height][CONFIG_T::out_width]) {
#pragma HLS array_partition variable=feature_in  complete dim=2
#pragma HLS array_partition variable=feature_in  complete dim=3
#pragma HLS array_partition variable=feature_out complete dim=2
#pragma HLS array_partition variable=feature_out complete dim=3
	//pool
	for (int oh = 0; oh < CONFIG_T::out_height; oh++) {
		for (int ow = 0; ow < CONFIG_T::out_width; ow++) {
			for (int ic = 0; ic < CONFIG_T::n_chan; ic++) {
#pragma HLS pipeline
				typename CONFIG_T::accum_t row_avg[CONFIG_T::filt_height] = { 0 };
				for (int kh = 0; kh < CONFIG_T::filt_height; kh++) {
					row_avg[kh] = average7<CONFIG_T>(
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 0], 
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 1], 
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 2], 
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 3], 
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 4], 
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 5], 
						feature_in[ic][oh*CONFIG_T::stride_height + kh][ow*CONFIG_T::stride_width + 6] );
				}
				feature_out[ic][oh][ow] = (typename CONFIG_T::feature_t)average7<CONFIG_T>(row_avg[0], row_avg[1], row_avg[2], row_avg[3], row_avg[4], row_avg[5], row_avg[6]);
			}
		}
	}
};

} // end namespace

#endif