#ifndef NNET_BUFFER_H_
#define NNET_BUFFER_H_
#include<iostream>
#include <cstdlib>
#include "nnet_common.h"


namespace nnet{

    struct Feature_Memory {
        typedef float feature_type;
        static const unsigned channel    = 100;
        static const unsigned width      = 5;
        static const unsigned height     = 5;
    };

    struct Weight_Memory {
        typedef float weight_type;
        static const unsigned out_channel   = 100;
        static const unsigned in_channel    = 100;
        static const unsigned height        = 3;
        static const unsigned width         = 3;
    };

// We don't need to copy weights back to Global BRAM and DDR. 
// Some functions perform the same operations while only the names of the parameters are different from each other.
// These duplicate functions are still reserved, for convenience.
// TODO: update pragmas according to the design of PE.

// clear the buffer
	template<typename BUFFER_CONFIG>
	void clear_buffer(
		typename BUFFER_CONFIG::feature_type data[BUFFER_CONFIG::channel][BUFFER_CONFIG::height][BUFFER_CONFIG::width])
	{
		//std::cout << "clear buffer";
		for (int c_idx = 0; c_idx < BUFFER_CONFIG::channel; ++c_idx)
		{
			for (int h_idx = 0; h_idx < BUFFER_CONFIG::height; ++h_idx)
			{
#pragma HLS PIPELINE
				for (int w_idx = 0; w_idx < BUFFER_CONFIG::width; ++w_idx)
				{
#pragma HLS UNROLL
					//std::cout <<  "";
					data[c_idx][h_idx][w_idx]
						= (typename BUFFER_CONFIG::feature_type) 0;
				}
			}
		}
		std::cout << std::endl;
	} // end clear buffer



// copy weights from DDR to BRAM
template<typename SRC_CONFIG, typename DST_CONFIG>
void copy_weights_DDR2BRAM(
    typename SRC_CONFIG::weight_type src[SRC_CONFIG::out_channel*SRC_CONFIG::in_channel][SRC_CONFIG::height][SRC_CONFIG::width],
    typename DST_CONFIG::weight_type dst[DST_CONFIG::out_channel][DST_CONFIG::in_channel][DST_CONFIG::height][DST_CONFIG::width],
    int out_offset, int out_c_num,
    int in_offset, int in_c_num)
    {
    //std::cout << "in copy_weights_DDR2BRAM" << std::endl;
        for(int o_idx = 0; o_idx < out_c_num; ++ o_idx)
        {
            for(int i_idx = 0; i_idx < in_c_num; ++ i_idx)
            {
                for(int h_idx = 0; h_idx < DST_CONFIG::height; ++ h_idx)
                {
#pragma HLS PIPELINE
                    for(int w_idx = 0; w_idx < DST_CONFIG::width; ++ w_idx)
                    {
#pragma HLS UNROLL
                        dst[o_idx][i_idx][h_idx][w_idx] 
                            = (typename DST_CONFIG::weight_type)(src[(o_idx + out_offset) *SRC_CONFIG::in_channel+i_idx + in_offset][h_idx][w_idx]);
                        //std::cout << dst[o_idx][i_idx][h_idx][w_idx] << ",";
                    }
                }
            }
        }
		//std::cout <<  std::endl;
    } // end copy_weights_DDR2BRAM

// copy weights from global BRAM to local BRAM
template<typename SRC_CONFIG, typename DST_CONFIG>
void copy_weights_g2l(
    typename SRC_CONFIG::weight_type src[SRC_CONFIG::out_channel][SRC_CONFIG::in_channel][SRC_CONFIG::height][SRC_CONFIG::width],
    typename DST_CONFIG::weight_type dst[DST_CONFIG::out_channel][DST_CONFIG::in_channel][DST_CONFIG::height][DST_CONFIG::width],
    int out_offset, int out_c_num,
    int in_offset, int in_c_num)
    {
	//std::cout << "in copy_weights_g2l" << std::endl;
        for(int o_idx = 0; o_idx < out_c_num; ++ o_idx)
        {
            for(int i_idx = 0; i_idx < in_c_num; ++ i_idx)
            {
                for(int h_idx = 0; h_idx < DST_CONFIG::height; ++ h_idx)
                {
#pragma HLS UNROLL
                    for(int w_idx = 0; w_idx < DST_CONFIG::width; ++ w_idx)
                    {
#pragma HLS UNROLL
                        dst[o_idx][i_idx][h_idx][w_idx] 
                            = (typename DST_CONFIG::weight_type)(src[o_idx + out_offset][i_idx + in_offset][h_idx][w_idx]);
						//std::cout << dst[o_idx][i_idx][h_idx][w_idx] << ",";
                    }
                }
            }
        }
		//std::cout << std::endl;
    } // end copy_weights_g2l


// copy features from DDR to Global BRAM
template<typename SRC_CONFIG, typename DST_CONFIG>
void copy_features_DDR2BRAM(
    typename SRC_CONFIG::feature_type src[SRC_CONFIG::channel*SRC_CONFIG::height*SRC_CONFIG::width],
    typename DST_CONFIG::feature_type dst[DST_CONFIG::channel][DST_CONFIG::height][DST_CONFIG::width],
    int src_c_offset,  int c_num,
    int src_h_offset,  int h_num,
    int src_w_offset,  int w_num)
    {
	//std::cout << "in copy_features_DDR2BRAM" << std::endl;
        for (int c_idx = 0; c_idx < c_num; c_idx++)
        {
            for (int h_idx = 0; h_idx < h_num; h_idx++)
            {
#pragma HLS PIPELINE
                for (int w_idx = 0; w_idx < w_num; w_idx++)
                {
                    dst[c_idx][h_idx][w_idx]
                        = (typename DST_CONFIG::feature_type)(src[(c_idx + src_c_offset) *SRC_CONFIG::height * SRC_CONFIG::width+(h_idx + src_h_offset) * SRC_CONFIG::width+w_idx + src_w_offset]);
					//std::cout << dst[c_idx][h_idx][w_idx] << ",";
                }
            }
        }
		//std::cout << std::endl;
    } // end copy_features_DDR2BRAM



// copy features from Global BRAM to DDR
template<typename SRC_CONFIG, typename DST_CONFIG>
void copy_features_BRAM2DDR(
    typename SRC_CONFIG::feature_type src[SRC_CONFIG::channel][SRC_CONFIG::height][SRC_CONFIG::width],
    typename DST_CONFIG::feature_type dst[DST_CONFIG::channel*DST_CONFIG::height*DST_CONFIG::width],
    int dst_c_offset, int c_num,
    int dst_h_offset, int h_num,
    int dst_w_offset, int w_num)
    {
	//std::cout << "in copy_features_BRAM2DDR" << std::endl;
        for (int c_idx = 0; c_idx < c_num; c_idx++)
        {
            for (int h_idx = 0; h_idx < h_num; h_idx++)
            {
#pragma HLS PIPELINE
                for (int w_idx = 0; w_idx < w_num; w_idx++)
                {
                    dst[(c_idx + dst_c_offset) * DST_CONFIG::height * DST_CONFIG::width+(h_idx + dst_h_offset) * DST_CONFIG::width +w_idx + dst_w_offset]
                        = (typename DST_CONFIG::feature_type)(src[c_idx][h_idx][w_idx]);
					//std::cout << dst[c_idx + dst_c_offset][h_idx + dst_h_offset][w_idx + dst_w_offset]<<",";
                }
            }
        }
		//std::cout << std::endl;
    } // end copy_features_BRAM2DDR

// copy features from Global BRAM to Local BRAM
// add padding here
template<typename SRC_CONFIG, typename DST_CONFIG>
void copy_features_g2l(
    typename SRC_CONFIG::feature_type src[SRC_CONFIG::channel][SRC_CONFIG::height][SRC_CONFIG::width],
    typename DST_CONFIG::feature_type dst[DST_CONFIG::channel][DST_CONFIG::height][DST_CONFIG::width],
    int src_c_offset, int dst_c_offset, int c_num,
    int src_h_offset, int dst_h_offset, int h_num,
    int src_w_offset, int dst_w_offset, int w_num)
    {
	//std::cout << "in copy_features_g2l" << std::endl;
        for (int c_idx = 0; c_idx < c_num; c_idx++)
        {
#pragma HLS PIPELINE
            for (int h_idx = 0; h_idx < h_num; h_idx++)
            {
#pragma HLS UNROLL
                for (int w_idx = 0; w_idx < w_num; w_idx++)
                {
#pragma HLS UNROLL
                    dst[c_idx+ dst_c_offset][h_idx+ dst_h_offset][w_idx+ dst_w_offset]
                        = (typename DST_CONFIG::feature_type)(src[c_idx + src_c_offset][h_idx + src_h_offset][w_idx + src_w_offset]);
					//std::cout << dst[c_idx + dst_c_offset][h_idx + dst_h_offset][w_idx + dst_w_offset] << ",";
                }
            }
        }
		//std::cout << std::endl;
    } // end copy_features_g2l

// copy features from Local BRAM to Global BRAM
template<typename SRC_CONFIG, typename DST_CONFIG>
void copy_features_l2g(
    typename SRC_CONFIG::feature_type src[SRC_CONFIG::channel][SRC_CONFIG::height][SRC_CONFIG::width],
    typename DST_CONFIG::feature_type dst[DST_CONFIG::channel][DST_CONFIG::height][DST_CONFIG::width],
    int dst_c_offset, int c_num,
    int dst_h_offset, int h_num,
    int dst_w_offset, int w_num)
    {
	//std::cout << "in copy_features_l2g" << std::endl;

        for (int c_idx = 0; c_idx < c_num; c_idx++)
        {
#pragma HLS PIPELINE
            for (int h_idx = 0; h_idx < h_num; h_idx++)
            {
#pragma HLS UNROLL
                for (int w_idx = 0; w_idx < w_num; w_idx++)
                {
#pragma HLS UNROLL
                    dst[c_idx + dst_c_offset][h_idx + dst_h_offset][w_idx + dst_w_offset]
                        = (typename DST_CONFIG::feature_type)(src[c_idx][h_idx][w_idx]);
					//std::cout << dst[c_idx + dst_c_offset][h_idx + dst_h_offset][w_idx + dst_w_offset] << ",";
                }
            }
        }
		//std::cout << std::endl;
    } // end copy_features_l2g

} // end namespace

#endif