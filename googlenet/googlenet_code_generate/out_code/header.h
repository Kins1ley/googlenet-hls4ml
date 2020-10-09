#ifndef HEADER_H_
#define HEADER_H_

#include <cstdlib>

#ifndef CPPSIM
	#include <ap_fixed.h>
	#include <ap_int.h>
	typedef ap_int<8>  FIX_INT8;
	typedef ap_int<16> FIX_INT20;
	typedef ap_int<32> FIX_INT32;
#else
	#include <iostream>
	typedef int FIX_INT8;
	typedef int FIX_INT20;
	typedef int FIX_INT32;
#endif

#define DIV_CEIL(x,y) ((x)/(y)+(((x)%(y)==0)?0:1))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

const int IMAGE_CH = 3; // image channel
const int IMAGE_H = 224;// image height
const int IMAGE_W = 224;// image width
const int OU = 1000;//number of final output

///config of DRAM with shared ports
const int DDR_WEIGHT_7x7_OUT_CHANNEL = 1024;
const int DDR_WEIGHT_7x7_IN_CHANNEL = 1024;
const int DDR_WEIGHT_5x5_OUT_CHANNEL = 1024;
const int DDR_WEIGHT_5x5_IN_CHANNEL = 1024;
const int DDR_WEIGHT_3x3_OUT_CHANNEL = 1024;
const int DDR_WEIGHT_3x3_IN_CHANNEL = 1024;
const int DDR_WEIGHT_1x1_OUT_CHANNEL = 1024;
const int DDR_WEIGHT_1x1_IN_CHANNEL = 1024;
const int DDR_BIAS_NUM = 1024;
///config of global BRAM
const int NUM_WEIGHT_GLOBAL_7x7 = 2; //number of global bram of weights
const int OUT_CHANNEL_WEIGHT_GLOBAL_7x7 = 32;//size of global bram of weights
const int IN_CHANNEL_WEIGHT_GLOBAL_7x7 = 3;//size of global bram of weights
const int NUM_WEIGHT_GLOBAL_5x5 = 8; //number of global bram of weights
const int OUT_CHANNEL_WEIGHT_GLOBAL_5x5 = 32;//size of global bram of weights
const int IN_CHANNEL_WEIGHT_GLOBAL_5x5 = 32;//size of global bram of weights
const int NUM_WEIGHT_GLOBAL_3x3 = 8; //number of global bram of weights
const int OUT_CHANNEL_WEIGHT_GLOBAL_3x3 = 32;//size of global bram of weights
const int IN_CHANNEL_WEIGHT_GLOBAL_3x3 = 32;//size of global bram of weights
const int NUM_WEIGHT_GLOBAL_1x1 = 8;//number of global bram of weights
const int OUT_CHANNEL_WEIGHT_GLOBAL_1x1 = 32;//size of global bram of weights
const int IN_CHANNEL_WEIGHT_GLOBAL_1x1 = 32;//size of global bram of weights

const int NUM_FEATURE_GLOBAL = 8;//number of global bram of feature
const int CHANNEL_FEATURE_GLOBAL = 64;//size of global bram of feature
const int WIDTH_FEATURE_GLOBAL = 63;//size of global bram of feature
const int HEIGHT_FEATURE_GLOBAL = 63;//size of global bram of feature

///config of local BRAM and PEs
//conv1x1_s1
const int NUM_PE_CONV1x1_S1 = 16;//number of local bram should be consistent with number of PEs
const int IN_CHAN_CONV1x1_S1 = 16;//in_channel of input feature and weight
const int OUT_CHAN_CONV1x1_S1 = 1;//out_channel of output feature and weight
const int OUT_HEIGHT_CONV1x1_S1 = 7;
const int OUT_WIDTH_CONV1x1_S1 = 7;
const int KERNEL_HEIGHT_CONV1x1_S1 = 1;
const int KERNEL_WIDTH_CONV1x1_S1 = 1;
const int STRIDE_CONV1x1_S1 = 1;
const int IN_HEIGHT_CONV1x1_S1 = (OUT_HEIGHT_CONV1x1_S1 - 1)*STRIDE_CONV1x1_S1 + KERNEL_HEIGHT_CONV1x1_S1;
const int IN_WIDTH_CONV1x1_S1 = (OUT_WIDTH_CONV1x1_S1 - 1)*STRIDE_CONV1x1_S1 + KERNEL_WIDTH_CONV1x1_S1;


//conv3x3_s1
const int NUM_PE_CONV3x3_S1 = 16;//number of local bram should be consistent with number of PEs
const int IN_CHAN_CONV3x3_S1 = 16;//in_channel of input feature and weight
const int OUT_CHAN_CONV3x3_S1 = 1;//out_channel of output feature and weight
const int OUT_HEIGHT_CONV3x3_S1 = 7;
const int OUT_WIDTH_CONV3x3_S1 = 7;
const int KERNEL_HEIGHT_CONV3x3_S1 = 3;
const int KERNEL_WIDTH_CONV3x3_S1 = 3;
const int STRIDE_CONV3x3_S1 = 1;
const int IN_HEIGHT_CONV3x3_S1 = (OUT_HEIGHT_CONV3x3_S1 - 1)*STRIDE_CONV3x3_S1 + KERNEL_HEIGHT_CONV3x3_S1;
const int IN_WIDTH_CONV3x3_S1 = (OUT_WIDTH_CONV3x3_S1 - 1)*STRIDE_CONV3x3_S1 + KERNEL_WIDTH_CONV3x3_S1;


//conv5x5_s1
const int NUM_PE_CONV5x5_S1 = 16;//number of local bram should be consistent with number of PEs
const int IN_CHAN_CONV5x5_S1 = 16;
const int OUT_CHAN_CONV5x5_S1 = 1;
const int OUT_HEIGHT_CONV5x5_S1 = 7;
const int OUT_WIDTH_CONV5x5_S1 = 7;
const int KERNEL_HEIGHT_CONV5x5_S1 = 5;
const int KERNEL_WIDTH_CONV5x5_S1 = 5;
const int STRIDE_CONV5x5_S1 = 1;
const int IN_HEIGHT_CONV5x5_S1 = (OUT_HEIGHT_CONV5x5_S1 - 1)*STRIDE_CONV5x5_S1 + KERNEL_HEIGHT_CONV5x5_S1;
const int IN_WIDTH_CONV5x5_S1 = (OUT_WIDTH_CONV5x5_S1 - 1)*STRIDE_CONV5x5_S1 + KERNEL_WIDTH_CONV5x5_S1;


//conv7x7_s2
const int NUM_PE_CONV7x7_S2 = 2;//number of local bram should be consistent with number of PEs
const int IN_CHAN_CONV7x7_S2 = 3;
const int OUT_CHAN_CONV7x7_S2 = 1;
const int OUT_HEIGHT_CONV7x7_S2 = 7;
const int OUT_WIDTH_CONV7x7_S2 = 7;
const int KERNEL_HEIGHT_CONV7x7_S2 = 7;
const int KERNEL_WIDTH_CONV7x7_S2 = 7;
const int STRIDE_CONV7x7_S2 = 2;
const int IN_HEIGHT_CONV7x7_S2 = (OUT_HEIGHT_CONV7x7_S2-1)* STRIDE_CONV7x7_S2 + KERNEL_HEIGHT_CONV7x7_S2;
const int IN_WIDTH_CONV7x7_S2 = (OUT_WIDTH_CONV7x7_S2-1)* STRIDE_CONV7x7_S2 + KERNEL_WIDTH_CONV7x7_S2;


///config of pooling operation
//maxpool3x3_s1
const int NUM_PE_MAXPOOL3x3_S1 = 2;
const int N_CHAN_MAXPOOL3x3_S1 = 16;
const int OUT_HEIGHT_MAXPOOL3x3_S1 = 7;
const int OUT_WIDTH_MAXPOOL3x3_S1 = 7;
const int KERNEL_HEIGHT_MAXPOOL3x3_S1 = 3;
const int KERNEL_WIDTH_MAXPOOL3x3_S1 = 3;
const int STRIDE_MAXPOOL3x3_S1 = 1;
const int IN_HEIGHT_MAXPOOL3x3_S1 = (OUT_HEIGHT_MAXPOOL3x3_S1 - 1)* STRIDE_MAXPOOL3x3_S1 + KERNEL_HEIGHT_MAXPOOL3x3_S1;
const int IN_WIDTH_MAXPOOL3x3_S1 = (OUT_WIDTH_MAXPOOL3x3_S1 - 1)* STRIDE_MAXPOOL3x3_S1 + KERNEL_WIDTH_MAXPOOL3x3_S1;

//maxpool3x3_s2
const int NUM_PE_MAXPOOL3x3_S2 = 2;
const int N_CHAN_MAXPOOL3x3_S2 = 16;
const int OUT_HEIGHT_MAXPOOL3x3_S2 = 7;
const int OUT_WIDTH_MAXPOOL3x3_S2 = 7;
const int KERNEL_HEIGHT_MAXPOOL3x3_S2 = 3;
const int KERNEL_WIDTH_MAXPOOL3x3_S2 = 3;
const int STRIDE_MAXPOOL3x3_S2 = 2;
const int IN_HEIGHT_MAXPOOL3x3_S2 = (OUT_HEIGHT_MAXPOOL3x3_S2-1)* STRIDE_MAXPOOL3x3_S2 + KERNEL_HEIGHT_MAXPOOL3x3_S2;
const int IN_WIDTH_MAXPOOL3x3_S2 = (OUT_WIDTH_MAXPOOL3x3_S2-1)* STRIDE_MAXPOOL3x3_S2 + KERNEL_WIDTH_MAXPOOL3x3_S2;

//avgpool7x7_s1
const int NUM_PE_AVGPOOL7x7_S1 = 2;
const int N_CHAN_AVGPOOL7x7_S1 = 16;
const int OUT_HEIGHT_AVGPOOL7x7_S1 = 1;
const int OUT_WIDTH_AVGPOOL7x7_S1 = 1;
const int KERNEL_HEIGHT_AVGPOOL7x7_S1 = 7;
const int KERNEL_WIDTH_AVGPOOL7x7_S1 = 7;
const int STRIDE_AVGPOOL7x7_S1 = 1;
const int IN_HEIGHT_AVGPOOL7x7_S1 = (OUT_HEIGHT_AVGPOOL7x7_S1 - 1)* STRIDE_AVGPOOL7x7_S1 + KERNEL_HEIGHT_AVGPOOL7x7_S1;
const int IN_WIDTH_AVGPOOL7x7_S1 = (OUT_WIDTH_AVGPOOL7x7_S1 - 1)* STRIDE_AVGPOOL7x7_S1 + KERNEL_WIDTH_AVGPOOL7x7_S1;


//config of layers
/////header_insert/////
///layer conv1_7x7_s2
const int conv1_7x7_s2_in_channel = 3;
const int conv1_7x7_s2_in_height = 224;
const int conv1_7x7_s2_in_width = 224;
const int conv1_7x7_s2_pad_top = 3;
const int conv1_7x7_s2_pad_bottom = 3;
const int conv1_7x7_s2_pad_left = 3;
const int conv1_7x7_s2_pad_right = 3;
const int conv1_7x7_s2_kernel_num = 64;
const int conv1_7x7_s2_kernel_channel = conv1_7x7_s2_in_channel;
const int conv1_7x7_s2_kernel_channel_DDR_offset = 0;
const int conv1_7x7_s2_kernel_height = 7;
const int conv1_7x7_s2_kernel_width = 7;
const int conv1_7x7_s2_bias_num = conv1_7x7_s2_kernel_num;
const int conv1_7x7_s2_bias_DDR_offset = 0;
const int conv1_7x7_s2_out_channel = conv1_7x7_s2_kernel_num;
const int conv1_7x7_s2_out_height = 112;
const int conv1_7x7_s2_out_width = 112;
const int conv1_7x7_s2_stride = 2;
const int conv1_7x7_s2_out_channel_DDR_offset = 0;

///layer pool1_3x3_s2
const int pool1_3x3_s2_in_channel = 64;
const int pool1_3x3_s2_in_height = 112;
const int pool1_3x3_s2_in_width = 112;
const int pool1_3x3_s2_pad_top = 0;
const int pool1_3x3_s2_pad_bottom = 0;
const int pool1_3x3_s2_pad_left = 2;
const int pool1_3x3_s2_pad_right = 2;
const int pool1_3x3_s2_out_channel = pool1_3x3_s2_in_channel;
const int pool1_3x3_s2_kernel_height = 3;
const int pool1_3x3_s2_kernel_width = 3;
const int pool1_3x3_s2_out_height = 56;
const int pool1_3x3_s2_out_width = 56;
const int pool1_3x3_s2_stride = 2;
const int pool1_3x3_s2_out_channel_DDR_offset = 0;

///layer pool1_norm1
const int pool1_norm1_in_channel = 64;
const int pool1_norm1_in_height = 56;
const int pool1_norm1_in_width = 56;
const int pool1_norm1_out_channel = pool1_norm1_in_channel;
const int pool1_norm1_out_height = pool1_norm1_in_height;
const int pool1_norm1_out_width = pool1_norm1_in_width;
const int pool1_norm1_depth_radius = 5;

///layer conv2_3x3_reduce
const int conv2_3x3_reduce_in_channel = 64;
const int conv2_3x3_reduce_in_height = 56;
const int conv2_3x3_reduce_in_width = 56;
const int conv2_3x3_reduce_pad_top = 0;
const int conv2_3x3_reduce_pad_bottom = 0;
const int conv2_3x3_reduce_pad_left = 0;
const int conv2_3x3_reduce_pad_right = 0;
const int conv2_3x3_reduce_kernel_num = 64;
const int conv2_3x3_reduce_kernel_channel = conv2_3x3_reduce_in_channel;
const int conv2_3x3_reduce_kernel_channel_DDR_offset = 0;
const int conv2_3x3_reduce_kernel_height = 1;
const int conv2_3x3_reduce_kernel_width = 1;
const int conv2_3x3_reduce_bias_num = conv2_3x3_reduce_kernel_num;
const int conv2_3x3_reduce_bias_DDR_offset = 0;
const int conv2_3x3_reduce_out_channel = conv2_3x3_reduce_kernel_num;
const int conv2_3x3_reduce_out_height = 56;
const int conv2_3x3_reduce_out_width = 56;
const int conv2_3x3_reduce_stride = 1;
const int conv2_3x3_reduce_out_channel_DDR_offset = 0;

///layer conv2_3x3
const int conv2_3x3_in_channel = 64;
const int conv2_3x3_in_height = 56;
const int conv2_3x3_in_width = 56;
const int conv2_3x3_pad_top = 1;
const int conv2_3x3_pad_bottom = 1;
const int conv2_3x3_pad_left = 1;
const int conv2_3x3_pad_right = 1;
const int conv2_3x3_kernel_num = 192;
const int conv2_3x3_kernel_channel = conv2_3x3_in_channel;
const int conv2_3x3_kernel_channel_DDR_offset = 0;
const int conv2_3x3_kernel_height = 3;
const int conv2_3x3_kernel_width = 3;
const int conv2_3x3_bias_num = conv2_3x3_kernel_num;
const int conv2_3x3_bias_DDR_offset = 0;
const int conv2_3x3_out_channel = conv2_3x3_kernel_num;
const int conv2_3x3_out_height = 56;
const int conv2_3x3_out_width = 56;
const int conv2_3x3_stride = 1;
const int conv2_3x3_out_channel_DDR_offset = 0;

///layer conv2_norm2
const int conv2_norm2_in_channel = 192;
const int conv2_norm2_in_height = 56;
const int conv2_norm2_in_width = 56;
const int conv2_norm2_out_channel = conv2_norm2_in_channel;
const int conv2_norm2_out_height = conv2_norm2_in_height;
const int conv2_norm2_out_width = conv2_norm2_in_width;
const int conv2_norm2_depth_radius = 5;

///layer pool2_3x3_s2
const int pool2_3x3_s2_in_channel = 192;
const int pool2_3x3_s2_in_height = 56;
const int pool2_3x3_s2_in_width = 56;
const int pool2_3x3_s2_pad_top = 0;
const int pool2_3x3_s2_pad_bottom = 0;
const int pool2_3x3_s2_pad_left = 2;
const int pool2_3x3_s2_pad_right = 2;
const int pool2_3x3_s2_out_channel = pool2_3x3_s2_in_channel;
const int pool2_3x3_s2_kernel_height = 3;
const int pool2_3x3_s2_kernel_width = 3;
const int pool2_3x3_s2_out_height = 28;
const int pool2_3x3_s2_out_width = 28;
const int pool2_3x3_s2_stride = 2;
const int pool2_3x3_s2_out_channel_DDR_offset = 0;

///layer inception_3a_1x1
const int inception_3a_1x1_in_channel = 192;
const int inception_3a_1x1_in_height = 28;
const int inception_3a_1x1_in_width = 28;
const int inception_3a_1x1_pad_top = 0;
const int inception_3a_1x1_pad_bottom = 0;
const int inception_3a_1x1_pad_left = 0;
const int inception_3a_1x1_pad_right = 0;
const int inception_3a_1x1_kernel_num = 64;
const int inception_3a_1x1_kernel_channel = inception_3a_1x1_in_channel;
const int inception_3a_1x1_kernel_channel_DDR_offset = 0;
const int inception_3a_1x1_kernel_height = 1;
const int inception_3a_1x1_kernel_width = 1;
const int inception_3a_1x1_bias_num = inception_3a_1x1_kernel_num;
const int inception_3a_1x1_bias_DDR_offset = 0;
const int inception_3a_1x1_out_channel = inception_3a_1x1_kernel_num;
const int inception_3a_1x1_out_height = 28;
const int inception_3a_1x1_out_width = 28;
const int inception_3a_1x1_stride = 1;
const int inception_3a_1x1_out_channel_DDR_offset = 0;

///layer inception_3a_3x3_reduce
const int inception_3a_3x3_reduce_in_channel = 192;
const int inception_3a_3x3_reduce_in_height = 28;
const int inception_3a_3x3_reduce_in_width = 28;
const int inception_3a_3x3_reduce_pad_top = 0;
const int inception_3a_3x3_reduce_pad_bottom = 0;
const int inception_3a_3x3_reduce_pad_left = 0;
const int inception_3a_3x3_reduce_pad_right = 0;
const int inception_3a_3x3_reduce_kernel_num = 96;
const int inception_3a_3x3_reduce_kernel_channel = inception_3a_3x3_reduce_in_channel;
const int inception_3a_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_3a_3x3_reduce_kernel_height = 1;
const int inception_3a_3x3_reduce_kernel_width = 1;
const int inception_3a_3x3_reduce_bias_num = inception_3a_3x3_reduce_kernel_num;
const int inception_3a_3x3_reduce_bias_DDR_offset = 0;
const int inception_3a_3x3_reduce_out_channel = inception_3a_3x3_reduce_kernel_num;
const int inception_3a_3x3_reduce_out_height = 28;
const int inception_3a_3x3_reduce_out_width = 28;
const int inception_3a_3x3_reduce_stride = 1;
const int inception_3a_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_3a_3x3
const int inception_3a_3x3_in_channel = 96;
const int inception_3a_3x3_in_height = 28;
const int inception_3a_3x3_in_width = 28;
const int inception_3a_3x3_pad_top = 1;
const int inception_3a_3x3_pad_bottom = 1;
const int inception_3a_3x3_pad_left = 1;
const int inception_3a_3x3_pad_right = 1;
const int inception_3a_3x3_kernel_num = 128;
const int inception_3a_3x3_kernel_channel = inception_3a_3x3_in_channel;
const int inception_3a_3x3_kernel_channel_DDR_offset = 0;
const int inception_3a_3x3_kernel_height = 3;
const int inception_3a_3x3_kernel_width = 3;
const int inception_3a_3x3_bias_num = inception_3a_3x3_kernel_num;
const int inception_3a_3x3_bias_DDR_offset = 0;
const int inception_3a_3x3_out_channel = inception_3a_3x3_kernel_num;
const int inception_3a_3x3_out_height = 28;
const int inception_3a_3x3_out_width = 28;
const int inception_3a_3x3_stride = 1;
const int inception_3a_3x3_out_channel_DDR_offset = 0;

///layer inception_3a_5x5_reduce
const int inception_3a_5x5_reduce_in_channel = 192;
const int inception_3a_5x5_reduce_in_height = 28;
const int inception_3a_5x5_reduce_in_width = 28;
const int inception_3a_5x5_reduce_pad_top = 0;
const int inception_3a_5x5_reduce_pad_bottom = 0;
const int inception_3a_5x5_reduce_pad_left = 0;
const int inception_3a_5x5_reduce_pad_right = 0;
const int inception_3a_5x5_reduce_kernel_num = 16;
const int inception_3a_5x5_reduce_kernel_channel = inception_3a_5x5_reduce_in_channel;
const int inception_3a_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_3a_5x5_reduce_kernel_height = 1;
const int inception_3a_5x5_reduce_kernel_width = 1;
const int inception_3a_5x5_reduce_bias_num = inception_3a_5x5_reduce_kernel_num;
const int inception_3a_5x5_reduce_bias_DDR_offset = 0;
const int inception_3a_5x5_reduce_out_channel = inception_3a_5x5_reduce_kernel_num;
const int inception_3a_5x5_reduce_out_height = 28;
const int inception_3a_5x5_reduce_out_width = 28;
const int inception_3a_5x5_reduce_stride = 1;
const int inception_3a_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_3a_5x5
const int inception_3a_5x5_in_channel = 16;
const int inception_3a_5x5_in_height = 28;
const int inception_3a_5x5_in_width = 28;
const int inception_3a_5x5_pad_top = 2;
const int inception_3a_5x5_pad_bottom = 2;
const int inception_3a_5x5_pad_left = 2;
const int inception_3a_5x5_pad_right = 2;
const int inception_3a_5x5_kernel_num = 32;
const int inception_3a_5x5_kernel_channel = inception_3a_5x5_in_channel;
const int inception_3a_5x5_kernel_channel_DDR_offset = 0;
const int inception_3a_5x5_kernel_height = 5;
const int inception_3a_5x5_kernel_width = 5;
const int inception_3a_5x5_bias_num = inception_3a_5x5_kernel_num;
const int inception_3a_5x5_bias_DDR_offset = 0;
const int inception_3a_5x5_out_channel = inception_3a_5x5_kernel_num;
const int inception_3a_5x5_out_height = 28;
const int inception_3a_5x5_out_width = 28;
const int inception_3a_5x5_stride = 1;
const int inception_3a_5x5_out_channel_DDR_offset = 0;

///layer inception_3a_pool
const int inception_3a_pool_in_channel = 192;
const int inception_3a_pool_in_height = 28;
const int inception_3a_pool_in_width = 28;
const int inception_3a_pool_pad_top = 1;
const int inception_3a_pool_pad_bottom = 1;
const int inception_3a_pool_pad_left = 1;
const int inception_3a_pool_pad_right = 1;
const int inception_3a_pool_out_channel = inception_3a_pool_in_channel;
const int inception_3a_pool_kernel_height = 3;
const int inception_3a_pool_kernel_width = 3;
const int inception_3a_pool_out_height = 28;
const int inception_3a_pool_out_width = 28;
const int inception_3a_pool_stride = 1;
const int inception_3a_pool_out_channel_DDR_offset = 0;

///layer inception_3a_pool_proj
const int inception_3a_pool_proj_in_channel = 192;
const int inception_3a_pool_proj_in_height = 28;
const int inception_3a_pool_proj_in_width = 28;
const int inception_3a_pool_proj_pad_top = 0;
const int inception_3a_pool_proj_pad_bottom = 0;
const int inception_3a_pool_proj_pad_left = 0;
const int inception_3a_pool_proj_pad_right = 0;
const int inception_3a_pool_proj_kernel_num = 32;
const int inception_3a_pool_proj_kernel_channel = inception_3a_pool_proj_in_channel;
const int inception_3a_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_3a_pool_proj_kernel_height = 1;
const int inception_3a_pool_proj_kernel_width = 1;
const int inception_3a_pool_proj_bias_num = inception_3a_pool_proj_kernel_num;
const int inception_3a_pool_proj_bias_DDR_offset = 0;
const int inception_3a_pool_proj_out_channel = inception_3a_pool_proj_kernel_num;
const int inception_3a_pool_proj_out_height = 28;
const int inception_3a_pool_proj_out_width = 28;
const int inception_3a_pool_proj_stride = 1;
const int inception_3a_pool_proj_out_channel_DDR_offset = 0;

///layer inception_3b_1x1
const int inception_3b_1x1_in_channel = 256;
const int inception_3b_1x1_in_height = 28;
const int inception_3b_1x1_in_width = 28;
const int inception_3b_1x1_pad_top = 0;
const int inception_3b_1x1_pad_bottom = 0;
const int inception_3b_1x1_pad_left = 0;
const int inception_3b_1x1_pad_right = 0;
const int inception_3b_1x1_kernel_num = 128;
const int inception_3b_1x1_kernel_channel = inception_3b_1x1_in_channel;
const int inception_3b_1x1_kernel_channel_DDR_offset = 0;
const int inception_3b_1x1_kernel_height = 1;
const int inception_3b_1x1_kernel_width = 1;
const int inception_3b_1x1_bias_num = inception_3b_1x1_kernel_num;
const int inception_3b_1x1_bias_DDR_offset = 0;
const int inception_3b_1x1_out_channel = inception_3b_1x1_kernel_num;
const int inception_3b_1x1_out_height = 28;
const int inception_3b_1x1_out_width = 28;
const int inception_3b_1x1_stride = 1;
const int inception_3b_1x1_out_channel_DDR_offset = 0;

///layer inception_3b_3x3_reduce
const int inception_3b_3x3_reduce_in_channel = 256;
const int inception_3b_3x3_reduce_in_height = 28;
const int inception_3b_3x3_reduce_in_width = 28;
const int inception_3b_3x3_reduce_pad_top = 0;
const int inception_3b_3x3_reduce_pad_bottom = 0;
const int inception_3b_3x3_reduce_pad_left = 0;
const int inception_3b_3x3_reduce_pad_right = 0;
const int inception_3b_3x3_reduce_kernel_num = 128;
const int inception_3b_3x3_reduce_kernel_channel = inception_3b_3x3_reduce_in_channel;
const int inception_3b_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_3b_3x3_reduce_kernel_height = 1;
const int inception_3b_3x3_reduce_kernel_width = 1;
const int inception_3b_3x3_reduce_bias_num = inception_3b_3x3_reduce_kernel_num;
const int inception_3b_3x3_reduce_bias_DDR_offset = 0;
const int inception_3b_3x3_reduce_out_channel = inception_3b_3x3_reduce_kernel_num;
const int inception_3b_3x3_reduce_out_height = 28;
const int inception_3b_3x3_reduce_out_width = 28;
const int inception_3b_3x3_reduce_stride = 1;
const int inception_3b_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_3b_3x3
const int inception_3b_3x3_in_channel = 128;
const int inception_3b_3x3_in_height = 28;
const int inception_3b_3x3_in_width = 28;
const int inception_3b_3x3_pad_top = 1;
const int inception_3b_3x3_pad_bottom = 1;
const int inception_3b_3x3_pad_left = 1;
const int inception_3b_3x3_pad_right = 1;
const int inception_3b_3x3_kernel_num = 192;
const int inception_3b_3x3_kernel_channel = inception_3b_3x3_in_channel;
const int inception_3b_3x3_kernel_channel_DDR_offset = 0;
const int inception_3b_3x3_kernel_height = 3;
const int inception_3b_3x3_kernel_width = 3;
const int inception_3b_3x3_bias_num = inception_3b_3x3_kernel_num;
const int inception_3b_3x3_bias_DDR_offset = 0;
const int inception_3b_3x3_out_channel = inception_3b_3x3_kernel_num;
const int inception_3b_3x3_out_height = 28;
const int inception_3b_3x3_out_width = 28;
const int inception_3b_3x3_stride = 1;
const int inception_3b_3x3_out_channel_DDR_offset = 0;

///layer inception_3b_5x5_reduce
const int inception_3b_5x5_reduce_in_channel = 256;
const int inception_3b_5x5_reduce_in_height = 28;
const int inception_3b_5x5_reduce_in_width = 28;
const int inception_3b_5x5_reduce_pad_top = 0;
const int inception_3b_5x5_reduce_pad_bottom = 0;
const int inception_3b_5x5_reduce_pad_left = 0;
const int inception_3b_5x5_reduce_pad_right = 0;
const int inception_3b_5x5_reduce_kernel_num = 32;
const int inception_3b_5x5_reduce_kernel_channel = inception_3b_5x5_reduce_in_channel;
const int inception_3b_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_3b_5x5_reduce_kernel_height = 1;
const int inception_3b_5x5_reduce_kernel_width = 1;
const int inception_3b_5x5_reduce_bias_num = inception_3b_5x5_reduce_kernel_num;
const int inception_3b_5x5_reduce_bias_DDR_offset = 0;
const int inception_3b_5x5_reduce_out_channel = inception_3b_5x5_reduce_kernel_num;
const int inception_3b_5x5_reduce_out_height = 28;
const int inception_3b_5x5_reduce_out_width = 28;
const int inception_3b_5x5_reduce_stride = 1;
const int inception_3b_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_3b_5x5
const int inception_3b_5x5_in_channel = 32;
const int inception_3b_5x5_in_height = 28;
const int inception_3b_5x5_in_width = 28;
const int inception_3b_5x5_pad_top = 2;
const int inception_3b_5x5_pad_bottom = 2;
const int inception_3b_5x5_pad_left = 2;
const int inception_3b_5x5_pad_right = 2;
const int inception_3b_5x5_kernel_num = 96;
const int inception_3b_5x5_kernel_channel = inception_3b_5x5_in_channel;
const int inception_3b_5x5_kernel_channel_DDR_offset = 0;
const int inception_3b_5x5_kernel_height = 5;
const int inception_3b_5x5_kernel_width = 5;
const int inception_3b_5x5_bias_num = inception_3b_5x5_kernel_num;
const int inception_3b_5x5_bias_DDR_offset = 0;
const int inception_3b_5x5_out_channel = inception_3b_5x5_kernel_num;
const int inception_3b_5x5_out_height = 28;
const int inception_3b_5x5_out_width = 28;
const int inception_3b_5x5_stride = 1;
const int inception_3b_5x5_out_channel_DDR_offset = 0;

///layer inception_3b_pool
const int inception_3b_pool_in_channel = 256;
const int inception_3b_pool_in_height = 28;
const int inception_3b_pool_in_width = 28;
const int inception_3b_pool_pad_top = 1;
const int inception_3b_pool_pad_bottom = 1;
const int inception_3b_pool_pad_left = 1;
const int inception_3b_pool_pad_right = 1;
const int inception_3b_pool_out_channel = inception_3b_pool_in_channel;
const int inception_3b_pool_kernel_height = 3;
const int inception_3b_pool_kernel_width = 3;
const int inception_3b_pool_out_height = 28;
const int inception_3b_pool_out_width = 28;
const int inception_3b_pool_stride = 1;
const int inception_3b_pool_out_channel_DDR_offset = 0;

///layer inception_3b_pool_proj
const int inception_3b_pool_proj_in_channel = 256;
const int inception_3b_pool_proj_in_height = 28;
const int inception_3b_pool_proj_in_width = 28;
const int inception_3b_pool_proj_pad_top = 0;
const int inception_3b_pool_proj_pad_bottom = 0;
const int inception_3b_pool_proj_pad_left = 0;
const int inception_3b_pool_proj_pad_right = 0;
const int inception_3b_pool_proj_kernel_num = 64;
const int inception_3b_pool_proj_kernel_channel = inception_3b_pool_proj_in_channel;
const int inception_3b_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_3b_pool_proj_kernel_height = 1;
const int inception_3b_pool_proj_kernel_width = 1;
const int inception_3b_pool_proj_bias_num = inception_3b_pool_proj_kernel_num;
const int inception_3b_pool_proj_bias_DDR_offset = 0;
const int inception_3b_pool_proj_out_channel = inception_3b_pool_proj_kernel_num;
const int inception_3b_pool_proj_out_height = 28;
const int inception_3b_pool_proj_out_width = 28;
const int inception_3b_pool_proj_stride = 1;
const int inception_3b_pool_proj_out_channel_DDR_offset = 0;

///layer pool3_3x3_s2
const int pool3_3x3_s2_in_channel = 480;
const int pool3_3x3_s2_in_height = 28;
const int pool3_3x3_s2_in_width = 28;
const int pool3_3x3_s2_pad_top = 0;
const int pool3_3x3_s2_pad_bottom = 0;
const int pool3_3x3_s2_pad_left = 2;
const int pool3_3x3_s2_pad_right = 2;
const int pool3_3x3_s2_out_channel = pool3_3x3_s2_in_channel;
const int pool3_3x3_s2_kernel_height = 3;
const int pool3_3x3_s2_kernel_width = 3;
const int pool3_3x3_s2_out_height = 14;
const int pool3_3x3_s2_out_width = 14;
const int pool3_3x3_s2_stride = 2;
const int pool3_3x3_s2_out_channel_DDR_offset = 0;

///layer inception_4a_1x1
const int inception_4a_1x1_in_channel = 480;
const int inception_4a_1x1_in_height = 14;
const int inception_4a_1x1_in_width = 14;
const int inception_4a_1x1_pad_top = 0;
const int inception_4a_1x1_pad_bottom = 0;
const int inception_4a_1x1_pad_left = 0;
const int inception_4a_1x1_pad_right = 0;
const int inception_4a_1x1_kernel_num = 192;
const int inception_4a_1x1_kernel_channel = inception_4a_1x1_in_channel;
const int inception_4a_1x1_kernel_channel_DDR_offset = 0;
const int inception_4a_1x1_kernel_height = 1;
const int inception_4a_1x1_kernel_width = 1;
const int inception_4a_1x1_bias_num = inception_4a_1x1_kernel_num;
const int inception_4a_1x1_bias_DDR_offset = 0;
const int inception_4a_1x1_out_channel = inception_4a_1x1_kernel_num;
const int inception_4a_1x1_out_height = 14;
const int inception_4a_1x1_out_width = 14;
const int inception_4a_1x1_stride = 1;
const int inception_4a_1x1_out_channel_DDR_offset = 0;

///layer inception_4a_3x3_reduce
const int inception_4a_3x3_reduce_in_channel = 480;
const int inception_4a_3x3_reduce_in_height = 14;
const int inception_4a_3x3_reduce_in_width = 14;
const int inception_4a_3x3_reduce_pad_top = 0;
const int inception_4a_3x3_reduce_pad_bottom = 0;
const int inception_4a_3x3_reduce_pad_left = 0;
const int inception_4a_3x3_reduce_pad_right = 0;
const int inception_4a_3x3_reduce_kernel_num = 96;
const int inception_4a_3x3_reduce_kernel_channel = inception_4a_3x3_reduce_in_channel;
const int inception_4a_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_4a_3x3_reduce_kernel_height = 1;
const int inception_4a_3x3_reduce_kernel_width = 1;
const int inception_4a_3x3_reduce_bias_num = inception_4a_3x3_reduce_kernel_num;
const int inception_4a_3x3_reduce_bias_DDR_offset = 0;
const int inception_4a_3x3_reduce_out_channel = inception_4a_3x3_reduce_kernel_num;
const int inception_4a_3x3_reduce_out_height = 14;
const int inception_4a_3x3_reduce_out_width = 14;
const int inception_4a_3x3_reduce_stride = 1;
const int inception_4a_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_4a_3x3
const int inception_4a_3x3_in_channel = 96;
const int inception_4a_3x3_in_height = 14;
const int inception_4a_3x3_in_width = 14;
const int inception_4a_3x3_pad_top = 1;
const int inception_4a_3x3_pad_bottom = 1;
const int inception_4a_3x3_pad_left = 1;
const int inception_4a_3x3_pad_right = 1;
const int inception_4a_3x3_kernel_num = 208;
const int inception_4a_3x3_kernel_channel = inception_4a_3x3_in_channel;
const int inception_4a_3x3_kernel_channel_DDR_offset = 0;
const int inception_4a_3x3_kernel_height = 3;
const int inception_4a_3x3_kernel_width = 3;
const int inception_4a_3x3_bias_num = inception_4a_3x3_kernel_num;
const int inception_4a_3x3_bias_DDR_offset = 0;
const int inception_4a_3x3_out_channel = inception_4a_3x3_kernel_num;
const int inception_4a_3x3_out_height = 14;
const int inception_4a_3x3_out_width = 14;
const int inception_4a_3x3_stride = 1;
const int inception_4a_3x3_out_channel_DDR_offset = 0;

///layer inception_4a_5x5_reduce
const int inception_4a_5x5_reduce_in_channel = 480;
const int inception_4a_5x5_reduce_in_height = 14;
const int inception_4a_5x5_reduce_in_width = 14;
const int inception_4a_5x5_reduce_pad_top = 0;
const int inception_4a_5x5_reduce_pad_bottom = 0;
const int inception_4a_5x5_reduce_pad_left = 0;
const int inception_4a_5x5_reduce_pad_right = 0;
const int inception_4a_5x5_reduce_kernel_num = 16;
const int inception_4a_5x5_reduce_kernel_channel = inception_4a_5x5_reduce_in_channel;
const int inception_4a_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_4a_5x5_reduce_kernel_height = 1;
const int inception_4a_5x5_reduce_kernel_width = 1;
const int inception_4a_5x5_reduce_bias_num = inception_4a_5x5_reduce_kernel_num;
const int inception_4a_5x5_reduce_bias_DDR_offset = 0;
const int inception_4a_5x5_reduce_out_channel = inception_4a_5x5_reduce_kernel_num;
const int inception_4a_5x5_reduce_out_height = 14;
const int inception_4a_5x5_reduce_out_width = 14;
const int inception_4a_5x5_reduce_stride = 1;
const int inception_4a_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_4a_5x5
const int inception_4a_5x5_in_channel = 16;
const int inception_4a_5x5_in_height = 14;
const int inception_4a_5x5_in_width = 14;
const int inception_4a_5x5_pad_top = 2;
const int inception_4a_5x5_pad_bottom = 2;
const int inception_4a_5x5_pad_left = 2;
const int inception_4a_5x5_pad_right = 2;
const int inception_4a_5x5_kernel_num = 48;
const int inception_4a_5x5_kernel_channel = inception_4a_5x5_in_channel;
const int inception_4a_5x5_kernel_channel_DDR_offset = 0;
const int inception_4a_5x5_kernel_height = 5;
const int inception_4a_5x5_kernel_width = 5;
const int inception_4a_5x5_bias_num = inception_4a_5x5_kernel_num;
const int inception_4a_5x5_bias_DDR_offset = 0;
const int inception_4a_5x5_out_channel = inception_4a_5x5_kernel_num;
const int inception_4a_5x5_out_height = 14;
const int inception_4a_5x5_out_width = 14;
const int inception_4a_5x5_stride = 1;
const int inception_4a_5x5_out_channel_DDR_offset = 0;

///layer inception_4a_pool
const int inception_4a_pool_in_channel = 480;
const int inception_4a_pool_in_height = 14;
const int inception_4a_pool_in_width = 14;
const int inception_4a_pool_pad_top = 1;
const int inception_4a_pool_pad_bottom = 1;
const int inception_4a_pool_pad_left = 1;
const int inception_4a_pool_pad_right = 1;
const int inception_4a_pool_out_channel = inception_4a_pool_in_channel;
const int inception_4a_pool_kernel_height = 3;
const int inception_4a_pool_kernel_width = 3;
const int inception_4a_pool_out_height = 14;
const int inception_4a_pool_out_width = 14;
const int inception_4a_pool_stride = 1;
const int inception_4a_pool_out_channel_DDR_offset = 0;

///layer inception_4a_pool_proj
const int inception_4a_pool_proj_in_channel = 480;
const int inception_4a_pool_proj_in_height = 14;
const int inception_4a_pool_proj_in_width = 14;
const int inception_4a_pool_proj_pad_top = 0;
const int inception_4a_pool_proj_pad_bottom = 0;
const int inception_4a_pool_proj_pad_left = 0;
const int inception_4a_pool_proj_pad_right = 0;
const int inception_4a_pool_proj_kernel_num = 64;
const int inception_4a_pool_proj_kernel_channel = inception_4a_pool_proj_in_channel;
const int inception_4a_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_4a_pool_proj_kernel_height = 1;
const int inception_4a_pool_proj_kernel_width = 1;
const int inception_4a_pool_proj_bias_num = inception_4a_pool_proj_kernel_num;
const int inception_4a_pool_proj_bias_DDR_offset = 0;
const int inception_4a_pool_proj_out_channel = inception_4a_pool_proj_kernel_num;
const int inception_4a_pool_proj_out_height = 14;
const int inception_4a_pool_proj_out_width = 14;
const int inception_4a_pool_proj_stride = 1;
const int inception_4a_pool_proj_out_channel_DDR_offset = 0;

///layer inception_4b_1x1
const int inception_4b_1x1_in_channel = 512;
const int inception_4b_1x1_in_height = 14;
const int inception_4b_1x1_in_width = 14;
const int inception_4b_1x1_pad_top = 0;
const int inception_4b_1x1_pad_bottom = 0;
const int inception_4b_1x1_pad_left = 0;
const int inception_4b_1x1_pad_right = 0;
const int inception_4b_1x1_kernel_num = 160;
const int inception_4b_1x1_kernel_channel = inception_4b_1x1_in_channel;
const int inception_4b_1x1_kernel_channel_DDR_offset = 0;
const int inception_4b_1x1_kernel_height = 1;
const int inception_4b_1x1_kernel_width = 1;
const int inception_4b_1x1_bias_num = inception_4b_1x1_kernel_num;
const int inception_4b_1x1_bias_DDR_offset = 0;
const int inception_4b_1x1_out_channel = inception_4b_1x1_kernel_num;
const int inception_4b_1x1_out_height = 14;
const int inception_4b_1x1_out_width = 14;
const int inception_4b_1x1_stride = 1;
const int inception_4b_1x1_out_channel_DDR_offset = 0;

///layer inception_4b_3x3_reduce
const int inception_4b_3x3_reduce_in_channel = 512;
const int inception_4b_3x3_reduce_in_height = 14;
const int inception_4b_3x3_reduce_in_width = 14;
const int inception_4b_3x3_reduce_pad_top = 0;
const int inception_4b_3x3_reduce_pad_bottom = 0;
const int inception_4b_3x3_reduce_pad_left = 0;
const int inception_4b_3x3_reduce_pad_right = 0;
const int inception_4b_3x3_reduce_kernel_num = 112;
const int inception_4b_3x3_reduce_kernel_channel = inception_4b_3x3_reduce_in_channel;
const int inception_4b_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_4b_3x3_reduce_kernel_height = 1;
const int inception_4b_3x3_reduce_kernel_width = 1;
const int inception_4b_3x3_reduce_bias_num = inception_4b_3x3_reduce_kernel_num;
const int inception_4b_3x3_reduce_bias_DDR_offset = 0;
const int inception_4b_3x3_reduce_out_channel = inception_4b_3x3_reduce_kernel_num;
const int inception_4b_3x3_reduce_out_height = 14;
const int inception_4b_3x3_reduce_out_width = 14;
const int inception_4b_3x3_reduce_stride = 1;
const int inception_4b_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_4b_3x3
const int inception_4b_3x3_in_channel = 112;
const int inception_4b_3x3_in_height = 14;
const int inception_4b_3x3_in_width = 14;
const int inception_4b_3x3_pad_top = 1;
const int inception_4b_3x3_pad_bottom = 1;
const int inception_4b_3x3_pad_left = 1;
const int inception_4b_3x3_pad_right = 1;
const int inception_4b_3x3_kernel_num = 224;
const int inception_4b_3x3_kernel_channel = inception_4b_3x3_in_channel;
const int inception_4b_3x3_kernel_channel_DDR_offset = 0;
const int inception_4b_3x3_kernel_height = 3;
const int inception_4b_3x3_kernel_width = 3;
const int inception_4b_3x3_bias_num = inception_4b_3x3_kernel_num;
const int inception_4b_3x3_bias_DDR_offset = 0;
const int inception_4b_3x3_out_channel = inception_4b_3x3_kernel_num;
const int inception_4b_3x3_out_height = 14;
const int inception_4b_3x3_out_width = 14;
const int inception_4b_3x3_stride = 1;
const int inception_4b_3x3_out_channel_DDR_offset = 0;

///layer inception_4b_5x5_reduce
const int inception_4b_5x5_reduce_in_channel = 512;
const int inception_4b_5x5_reduce_in_height = 14;
const int inception_4b_5x5_reduce_in_width = 14;
const int inception_4b_5x5_reduce_pad_top = 0;
const int inception_4b_5x5_reduce_pad_bottom = 0;
const int inception_4b_5x5_reduce_pad_left = 0;
const int inception_4b_5x5_reduce_pad_right = 0;
const int inception_4b_5x5_reduce_kernel_num = 24;
const int inception_4b_5x5_reduce_kernel_channel = inception_4b_5x5_reduce_in_channel;
const int inception_4b_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_4b_5x5_reduce_kernel_height = 1;
const int inception_4b_5x5_reduce_kernel_width = 1;
const int inception_4b_5x5_reduce_bias_num = inception_4b_5x5_reduce_kernel_num;
const int inception_4b_5x5_reduce_bias_DDR_offset = 0;
const int inception_4b_5x5_reduce_out_channel = inception_4b_5x5_reduce_kernel_num;
const int inception_4b_5x5_reduce_out_height = 14;
const int inception_4b_5x5_reduce_out_width = 14;
const int inception_4b_5x5_reduce_stride = 1;
const int inception_4b_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_4b_5x5
const int inception_4b_5x5_in_channel = 24;
const int inception_4b_5x5_in_height = 14;
const int inception_4b_5x5_in_width = 14;
const int inception_4b_5x5_pad_top = 2;
const int inception_4b_5x5_pad_bottom = 2;
const int inception_4b_5x5_pad_left = 2;
const int inception_4b_5x5_pad_right = 2;
const int inception_4b_5x5_kernel_num = 64;
const int inception_4b_5x5_kernel_channel = inception_4b_5x5_in_channel;
const int inception_4b_5x5_kernel_channel_DDR_offset = 0;
const int inception_4b_5x5_kernel_height = 5;
const int inception_4b_5x5_kernel_width = 5;
const int inception_4b_5x5_bias_num = inception_4b_5x5_kernel_num;
const int inception_4b_5x5_bias_DDR_offset = 0;
const int inception_4b_5x5_out_channel = inception_4b_5x5_kernel_num;
const int inception_4b_5x5_out_height = 14;
const int inception_4b_5x5_out_width = 14;
const int inception_4b_5x5_stride = 1;
const int inception_4b_5x5_out_channel_DDR_offset = 0;

///layer inception_4b_pool
const int inception_4b_pool_in_channel = 512;
const int inception_4b_pool_in_height = 14;
const int inception_4b_pool_in_width = 14;
const int inception_4b_pool_pad_top = 1;
const int inception_4b_pool_pad_bottom = 1;
const int inception_4b_pool_pad_left = 1;
const int inception_4b_pool_pad_right = 1;
const int inception_4b_pool_out_channel = inception_4b_pool_in_channel;
const int inception_4b_pool_kernel_height = 3;
const int inception_4b_pool_kernel_width = 3;
const int inception_4b_pool_out_height = 14;
const int inception_4b_pool_out_width = 14;
const int inception_4b_pool_stride = 1;
const int inception_4b_pool_out_channel_DDR_offset = 0;

///layer inception_4b_pool_proj
const int inception_4b_pool_proj_in_channel = 512;
const int inception_4b_pool_proj_in_height = 14;
const int inception_4b_pool_proj_in_width = 14;
const int inception_4b_pool_proj_pad_top = 0;
const int inception_4b_pool_proj_pad_bottom = 0;
const int inception_4b_pool_proj_pad_left = 0;
const int inception_4b_pool_proj_pad_right = 0;
const int inception_4b_pool_proj_kernel_num = 64;
const int inception_4b_pool_proj_kernel_channel = inception_4b_pool_proj_in_channel;
const int inception_4b_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_4b_pool_proj_kernel_height = 1;
const int inception_4b_pool_proj_kernel_width = 1;
const int inception_4b_pool_proj_bias_num = inception_4b_pool_proj_kernel_num;
const int inception_4b_pool_proj_bias_DDR_offset = 0;
const int inception_4b_pool_proj_out_channel = inception_4b_pool_proj_kernel_num;
const int inception_4b_pool_proj_out_height = 14;
const int inception_4b_pool_proj_out_width = 14;
const int inception_4b_pool_proj_stride = 1;
const int inception_4b_pool_proj_out_channel_DDR_offset = 0;

///layer inception_4c_1x1
const int inception_4c_1x1_in_channel = 512;
const int inception_4c_1x1_in_height = 14;
const int inception_4c_1x1_in_width = 14;
const int inception_4c_1x1_pad_top = 0;
const int inception_4c_1x1_pad_bottom = 0;
const int inception_4c_1x1_pad_left = 0;
const int inception_4c_1x1_pad_right = 0;
const int inception_4c_1x1_kernel_num = 128;
const int inception_4c_1x1_kernel_channel = inception_4c_1x1_in_channel;
const int inception_4c_1x1_kernel_channel_DDR_offset = 0;
const int inception_4c_1x1_kernel_height = 1;
const int inception_4c_1x1_kernel_width = 1;
const int inception_4c_1x1_bias_num = inception_4c_1x1_kernel_num;
const int inception_4c_1x1_bias_DDR_offset = 0;
const int inception_4c_1x1_out_channel = inception_4c_1x1_kernel_num;
const int inception_4c_1x1_out_height = 14;
const int inception_4c_1x1_out_width = 14;
const int inception_4c_1x1_stride = 1;
const int inception_4c_1x1_out_channel_DDR_offset = 0;

///layer inception_4c_3x3_reduce
const int inception_4c_3x3_reduce_in_channel = 512;
const int inception_4c_3x3_reduce_in_height = 14;
const int inception_4c_3x3_reduce_in_width = 14;
const int inception_4c_3x3_reduce_pad_top = 0;
const int inception_4c_3x3_reduce_pad_bottom = 0;
const int inception_4c_3x3_reduce_pad_left = 0;
const int inception_4c_3x3_reduce_pad_right = 0;
const int inception_4c_3x3_reduce_kernel_num = 128;
const int inception_4c_3x3_reduce_kernel_channel = inception_4c_3x3_reduce_in_channel;
const int inception_4c_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_4c_3x3_reduce_kernel_height = 1;
const int inception_4c_3x3_reduce_kernel_width = 1;
const int inception_4c_3x3_reduce_bias_num = inception_4c_3x3_reduce_kernel_num;
const int inception_4c_3x3_reduce_bias_DDR_offset = 0;
const int inception_4c_3x3_reduce_out_channel = inception_4c_3x3_reduce_kernel_num;
const int inception_4c_3x3_reduce_out_height = 14;
const int inception_4c_3x3_reduce_out_width = 14;
const int inception_4c_3x3_reduce_stride = 1;
const int inception_4c_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_4c_3x3
const int inception_4c_3x3_in_channel = 128;
const int inception_4c_3x3_in_height = 14;
const int inception_4c_3x3_in_width = 14;
const int inception_4c_3x3_pad_top = 1;
const int inception_4c_3x3_pad_bottom = 1;
const int inception_4c_3x3_pad_left = 1;
const int inception_4c_3x3_pad_right = 1;
const int inception_4c_3x3_kernel_num = 256;
const int inception_4c_3x3_kernel_channel = inception_4c_3x3_in_channel;
const int inception_4c_3x3_kernel_channel_DDR_offset = 0;
const int inception_4c_3x3_kernel_height = 3;
const int inception_4c_3x3_kernel_width = 3;
const int inception_4c_3x3_bias_num = inception_4c_3x3_kernel_num;
const int inception_4c_3x3_bias_DDR_offset = 0;
const int inception_4c_3x3_out_channel = inception_4c_3x3_kernel_num;
const int inception_4c_3x3_out_height = 14;
const int inception_4c_3x3_out_width = 14;
const int inception_4c_3x3_stride = 1;
const int inception_4c_3x3_out_channel_DDR_offset = 0;

///layer inception_4c_5x5_reduce
const int inception_4c_5x5_reduce_in_channel = 512;
const int inception_4c_5x5_reduce_in_height = 14;
const int inception_4c_5x5_reduce_in_width = 14;
const int inception_4c_5x5_reduce_pad_top = 0;
const int inception_4c_5x5_reduce_pad_bottom = 0;
const int inception_4c_5x5_reduce_pad_left = 0;
const int inception_4c_5x5_reduce_pad_right = 0;
const int inception_4c_5x5_reduce_kernel_num = 24;
const int inception_4c_5x5_reduce_kernel_channel = inception_4c_5x5_reduce_in_channel;
const int inception_4c_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_4c_5x5_reduce_kernel_height = 1;
const int inception_4c_5x5_reduce_kernel_width = 1;
const int inception_4c_5x5_reduce_bias_num = inception_4c_5x5_reduce_kernel_num;
const int inception_4c_5x5_reduce_bias_DDR_offset = 0;
const int inception_4c_5x5_reduce_out_channel = inception_4c_5x5_reduce_kernel_num;
const int inception_4c_5x5_reduce_out_height = 14;
const int inception_4c_5x5_reduce_out_width = 14;
const int inception_4c_5x5_reduce_stride = 1;
const int inception_4c_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_4c_5x5
const int inception_4c_5x5_in_channel = 24;
const int inception_4c_5x5_in_height = 14;
const int inception_4c_5x5_in_width = 14;
const int inception_4c_5x5_pad_top = 2;
const int inception_4c_5x5_pad_bottom = 2;
const int inception_4c_5x5_pad_left = 2;
const int inception_4c_5x5_pad_right = 2;
const int inception_4c_5x5_kernel_num = 64;
const int inception_4c_5x5_kernel_channel = inception_4c_5x5_in_channel;
const int inception_4c_5x5_kernel_channel_DDR_offset = 0;
const int inception_4c_5x5_kernel_height = 5;
const int inception_4c_5x5_kernel_width = 5;
const int inception_4c_5x5_bias_num = inception_4c_5x5_kernel_num;
const int inception_4c_5x5_bias_DDR_offset = 0;
const int inception_4c_5x5_out_channel = inception_4c_5x5_kernel_num;
const int inception_4c_5x5_out_height = 14;
const int inception_4c_5x5_out_width = 14;
const int inception_4c_5x5_stride = 1;
const int inception_4c_5x5_out_channel_DDR_offset = 0;

///layer inception_4c_pool
const int inception_4c_pool_in_channel = 512;
const int inception_4c_pool_in_height = 14;
const int inception_4c_pool_in_width = 14;
const int inception_4c_pool_pad_top = 1;
const int inception_4c_pool_pad_bottom = 1;
const int inception_4c_pool_pad_left = 1;
const int inception_4c_pool_pad_right = 1;
const int inception_4c_pool_out_channel = inception_4c_pool_in_channel;
const int inception_4c_pool_kernel_height = 3;
const int inception_4c_pool_kernel_width = 3;
const int inception_4c_pool_out_height = 14;
const int inception_4c_pool_out_width = 14;
const int inception_4c_pool_stride = 1;
const int inception_4c_pool_out_channel_DDR_offset = 0;

///layer inception_4c_pool_proj
const int inception_4c_pool_proj_in_channel = 512;
const int inception_4c_pool_proj_in_height = 14;
const int inception_4c_pool_proj_in_width = 14;
const int inception_4c_pool_proj_pad_top = 0;
const int inception_4c_pool_proj_pad_bottom = 0;
const int inception_4c_pool_proj_pad_left = 0;
const int inception_4c_pool_proj_pad_right = 0;
const int inception_4c_pool_proj_kernel_num = 64;
const int inception_4c_pool_proj_kernel_channel = inception_4c_pool_proj_in_channel;
const int inception_4c_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_4c_pool_proj_kernel_height = 1;
const int inception_4c_pool_proj_kernel_width = 1;
const int inception_4c_pool_proj_bias_num = inception_4c_pool_proj_kernel_num;
const int inception_4c_pool_proj_bias_DDR_offset = 0;
const int inception_4c_pool_proj_out_channel = inception_4c_pool_proj_kernel_num;
const int inception_4c_pool_proj_out_height = 14;
const int inception_4c_pool_proj_out_width = 14;
const int inception_4c_pool_proj_stride = 1;
const int inception_4c_pool_proj_out_channel_DDR_offset = 0;

///layer inception_4d_1x1
const int inception_4d_1x1_in_channel = 512;
const int inception_4d_1x1_in_height = 14;
const int inception_4d_1x1_in_width = 14;
const int inception_4d_1x1_pad_top = 0;
const int inception_4d_1x1_pad_bottom = 0;
const int inception_4d_1x1_pad_left = 0;
const int inception_4d_1x1_pad_right = 0;
const int inception_4d_1x1_kernel_num = 112;
const int inception_4d_1x1_kernel_channel = inception_4d_1x1_in_channel;
const int inception_4d_1x1_kernel_channel_DDR_offset = 0;
const int inception_4d_1x1_kernel_height = 1;
const int inception_4d_1x1_kernel_width = 1;
const int inception_4d_1x1_bias_num = inception_4d_1x1_kernel_num;
const int inception_4d_1x1_bias_DDR_offset = 0;
const int inception_4d_1x1_out_channel = inception_4d_1x1_kernel_num;
const int inception_4d_1x1_out_height = 14;
const int inception_4d_1x1_out_width = 14;
const int inception_4d_1x1_stride = 1;
const int inception_4d_1x1_out_channel_DDR_offset = 0;

///layer inception_4d_3x3_reduce
const int inception_4d_3x3_reduce_in_channel = 512;
const int inception_4d_3x3_reduce_in_height = 14;
const int inception_4d_3x3_reduce_in_width = 14;
const int inception_4d_3x3_reduce_pad_top = 0;
const int inception_4d_3x3_reduce_pad_bottom = 0;
const int inception_4d_3x3_reduce_pad_left = 0;
const int inception_4d_3x3_reduce_pad_right = 0;
const int inception_4d_3x3_reduce_kernel_num = 144;
const int inception_4d_3x3_reduce_kernel_channel = inception_4d_3x3_reduce_in_channel;
const int inception_4d_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_4d_3x3_reduce_kernel_height = 1;
const int inception_4d_3x3_reduce_kernel_width = 1;
const int inception_4d_3x3_reduce_bias_num = inception_4d_3x3_reduce_kernel_num;
const int inception_4d_3x3_reduce_bias_DDR_offset = 0;
const int inception_4d_3x3_reduce_out_channel = inception_4d_3x3_reduce_kernel_num;
const int inception_4d_3x3_reduce_out_height = 14;
const int inception_4d_3x3_reduce_out_width = 14;
const int inception_4d_3x3_reduce_stride = 1;
const int inception_4d_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_4d_3x3
const int inception_4d_3x3_in_channel = 144;
const int inception_4d_3x3_in_height = 14;
const int inception_4d_3x3_in_width = 14;
const int inception_4d_3x3_pad_top = 1;
const int inception_4d_3x3_pad_bottom = 1;
const int inception_4d_3x3_pad_left = 1;
const int inception_4d_3x3_pad_right = 1;
const int inception_4d_3x3_kernel_num = 288;
const int inception_4d_3x3_kernel_channel = inception_4d_3x3_in_channel;
const int inception_4d_3x3_kernel_channel_DDR_offset = 0;
const int inception_4d_3x3_kernel_height = 3;
const int inception_4d_3x3_kernel_width = 3;
const int inception_4d_3x3_bias_num = inception_4d_3x3_kernel_num;
const int inception_4d_3x3_bias_DDR_offset = 0;
const int inception_4d_3x3_out_channel = inception_4d_3x3_kernel_num;
const int inception_4d_3x3_out_height = 14;
const int inception_4d_3x3_out_width = 14;
const int inception_4d_3x3_stride = 1;
const int inception_4d_3x3_out_channel_DDR_offset = 0;

///layer inception_4d_5x5_reduce
const int inception_4d_5x5_reduce_in_channel = 512;
const int inception_4d_5x5_reduce_in_height = 14;
const int inception_4d_5x5_reduce_in_width = 14;
const int inception_4d_5x5_reduce_pad_top = 0;
const int inception_4d_5x5_reduce_pad_bottom = 0;
const int inception_4d_5x5_reduce_pad_left = 0;
const int inception_4d_5x5_reduce_pad_right = 0;
const int inception_4d_5x5_reduce_kernel_num = 32;
const int inception_4d_5x5_reduce_kernel_channel = inception_4d_5x5_reduce_in_channel;
const int inception_4d_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_4d_5x5_reduce_kernel_height = 1;
const int inception_4d_5x5_reduce_kernel_width = 1;
const int inception_4d_5x5_reduce_bias_num = inception_4d_5x5_reduce_kernel_num;
const int inception_4d_5x5_reduce_bias_DDR_offset = 0;
const int inception_4d_5x5_reduce_out_channel = inception_4d_5x5_reduce_kernel_num;
const int inception_4d_5x5_reduce_out_height = 14;
const int inception_4d_5x5_reduce_out_width = 14;
const int inception_4d_5x5_reduce_stride = 1;
const int inception_4d_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_4d_5x5
const int inception_4d_5x5_in_channel = 32;
const int inception_4d_5x5_in_height = 14;
const int inception_4d_5x5_in_width = 14;
const int inception_4d_5x5_pad_top = 2;
const int inception_4d_5x5_pad_bottom = 2;
const int inception_4d_5x5_pad_left = 2;
const int inception_4d_5x5_pad_right = 2;
const int inception_4d_5x5_kernel_num = 64;
const int inception_4d_5x5_kernel_channel = inception_4d_5x5_in_channel;
const int inception_4d_5x5_kernel_channel_DDR_offset = 0;
const int inception_4d_5x5_kernel_height = 5;
const int inception_4d_5x5_kernel_width = 5;
const int inception_4d_5x5_bias_num = inception_4d_5x5_kernel_num;
const int inception_4d_5x5_bias_DDR_offset = 0;
const int inception_4d_5x5_out_channel = inception_4d_5x5_kernel_num;
const int inception_4d_5x5_out_height = 14;
const int inception_4d_5x5_out_width = 14;
const int inception_4d_5x5_stride = 1;
const int inception_4d_5x5_out_channel_DDR_offset = 0;

///layer inception_4d_pool
const int inception_4d_pool_in_channel = 512;
const int inception_4d_pool_in_height = 14;
const int inception_4d_pool_in_width = 14;
const int inception_4d_pool_pad_top = 1;
const int inception_4d_pool_pad_bottom = 1;
const int inception_4d_pool_pad_left = 1;
const int inception_4d_pool_pad_right = 1;
const int inception_4d_pool_out_channel = inception_4d_pool_in_channel;
const int inception_4d_pool_kernel_height = 3;
const int inception_4d_pool_kernel_width = 3;
const int inception_4d_pool_out_height = 14;
const int inception_4d_pool_out_width = 14;
const int inception_4d_pool_stride = 1;
const int inception_4d_pool_out_channel_DDR_offset = 0;

///layer inception_4d_pool_proj
const int inception_4d_pool_proj_in_channel = 512;
const int inception_4d_pool_proj_in_height = 14;
const int inception_4d_pool_proj_in_width = 14;
const int inception_4d_pool_proj_pad_top = 0;
const int inception_4d_pool_proj_pad_bottom = 0;
const int inception_4d_pool_proj_pad_left = 0;
const int inception_4d_pool_proj_pad_right = 0;
const int inception_4d_pool_proj_kernel_num = 64;
const int inception_4d_pool_proj_kernel_channel = inception_4d_pool_proj_in_channel;
const int inception_4d_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_4d_pool_proj_kernel_height = 1;
const int inception_4d_pool_proj_kernel_width = 1;
const int inception_4d_pool_proj_bias_num = inception_4d_pool_proj_kernel_num;
const int inception_4d_pool_proj_bias_DDR_offset = 0;
const int inception_4d_pool_proj_out_channel = inception_4d_pool_proj_kernel_num;
const int inception_4d_pool_proj_out_height = 14;
const int inception_4d_pool_proj_out_width = 14;
const int inception_4d_pool_proj_stride = 1;
const int inception_4d_pool_proj_out_channel_DDR_offset = 0;

///layer inception_4e_1x1
const int inception_4e_1x1_in_channel = 528;
const int inception_4e_1x1_in_height = 14;
const int inception_4e_1x1_in_width = 14;
const int inception_4e_1x1_pad_top = 0;
const int inception_4e_1x1_pad_bottom = 0;
const int inception_4e_1x1_pad_left = 0;
const int inception_4e_1x1_pad_right = 0;
const int inception_4e_1x1_kernel_num = 256;
const int inception_4e_1x1_kernel_channel = inception_4e_1x1_in_channel;
const int inception_4e_1x1_kernel_channel_DDR_offset = 0;
const int inception_4e_1x1_kernel_height = 1;
const int inception_4e_1x1_kernel_width = 1;
const int inception_4e_1x1_bias_num = inception_4e_1x1_kernel_num;
const int inception_4e_1x1_bias_DDR_offset = 0;
const int inception_4e_1x1_out_channel = inception_4e_1x1_kernel_num;
const int inception_4e_1x1_out_height = 14;
const int inception_4e_1x1_out_width = 14;
const int inception_4e_1x1_stride = 1;
const int inception_4e_1x1_out_channel_DDR_offset = 0;

///layer inception_4e_3x3_reduce
const int inception_4e_3x3_reduce_in_channel = 528;
const int inception_4e_3x3_reduce_in_height = 14;
const int inception_4e_3x3_reduce_in_width = 14;
const int inception_4e_3x3_reduce_pad_top = 0;
const int inception_4e_3x3_reduce_pad_bottom = 0;
const int inception_4e_3x3_reduce_pad_left = 0;
const int inception_4e_3x3_reduce_pad_right = 0;
const int inception_4e_3x3_reduce_kernel_num = 160;
const int inception_4e_3x3_reduce_kernel_channel = inception_4e_3x3_reduce_in_channel;
const int inception_4e_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_4e_3x3_reduce_kernel_height = 1;
const int inception_4e_3x3_reduce_kernel_width = 1;
const int inception_4e_3x3_reduce_bias_num = inception_4e_3x3_reduce_kernel_num;
const int inception_4e_3x3_reduce_bias_DDR_offset = 0;
const int inception_4e_3x3_reduce_out_channel = inception_4e_3x3_reduce_kernel_num;
const int inception_4e_3x3_reduce_out_height = 14;
const int inception_4e_3x3_reduce_out_width = 14;
const int inception_4e_3x3_reduce_stride = 1;
const int inception_4e_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_4e_3x3
const int inception_4e_3x3_in_channel = 160;
const int inception_4e_3x3_in_height = 14;
const int inception_4e_3x3_in_width = 14;
const int inception_4e_3x3_pad_top = 1;
const int inception_4e_3x3_pad_bottom = 1;
const int inception_4e_3x3_pad_left = 1;
const int inception_4e_3x3_pad_right = 1;
const int inception_4e_3x3_kernel_num = 320;
const int inception_4e_3x3_kernel_channel = inception_4e_3x3_in_channel;
const int inception_4e_3x3_kernel_channel_DDR_offset = 0;
const int inception_4e_3x3_kernel_height = 3;
const int inception_4e_3x3_kernel_width = 3;
const int inception_4e_3x3_bias_num = inception_4e_3x3_kernel_num;
const int inception_4e_3x3_bias_DDR_offset = 0;
const int inception_4e_3x3_out_channel = inception_4e_3x3_kernel_num;
const int inception_4e_3x3_out_height = 14;
const int inception_4e_3x3_out_width = 14;
const int inception_4e_3x3_stride = 1;
const int inception_4e_3x3_out_channel_DDR_offset = 0;

///layer inception_4e_5x5_reduce
const int inception_4e_5x5_reduce_in_channel = 528;
const int inception_4e_5x5_reduce_in_height = 14;
const int inception_4e_5x5_reduce_in_width = 14;
const int inception_4e_5x5_reduce_pad_top = 0;
const int inception_4e_5x5_reduce_pad_bottom = 0;
const int inception_4e_5x5_reduce_pad_left = 0;
const int inception_4e_5x5_reduce_pad_right = 0;
const int inception_4e_5x5_reduce_kernel_num = 32;
const int inception_4e_5x5_reduce_kernel_channel = inception_4e_5x5_reduce_in_channel;
const int inception_4e_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_4e_5x5_reduce_kernel_height = 1;
const int inception_4e_5x5_reduce_kernel_width = 1;
const int inception_4e_5x5_reduce_bias_num = inception_4e_5x5_reduce_kernel_num;
const int inception_4e_5x5_reduce_bias_DDR_offset = 0;
const int inception_4e_5x5_reduce_out_channel = inception_4e_5x5_reduce_kernel_num;
const int inception_4e_5x5_reduce_out_height = 14;
const int inception_4e_5x5_reduce_out_width = 14;
const int inception_4e_5x5_reduce_stride = 1;
const int inception_4e_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_4e_5x5
const int inception_4e_5x5_in_channel = 32;
const int inception_4e_5x5_in_height = 14;
const int inception_4e_5x5_in_width = 14;
const int inception_4e_5x5_pad_top = 2;
const int inception_4e_5x5_pad_bottom = 2;
const int inception_4e_5x5_pad_left = 2;
const int inception_4e_5x5_pad_right = 2;
const int inception_4e_5x5_kernel_num = 128;
const int inception_4e_5x5_kernel_channel = inception_4e_5x5_in_channel;
const int inception_4e_5x5_kernel_channel_DDR_offset = 0;
const int inception_4e_5x5_kernel_height = 5;
const int inception_4e_5x5_kernel_width = 5;
const int inception_4e_5x5_bias_num = inception_4e_5x5_kernel_num;
const int inception_4e_5x5_bias_DDR_offset = 0;
const int inception_4e_5x5_out_channel = inception_4e_5x5_kernel_num;
const int inception_4e_5x5_out_height = 14;
const int inception_4e_5x5_out_width = 14;
const int inception_4e_5x5_stride = 1;
const int inception_4e_5x5_out_channel_DDR_offset = 0;

///layer inception_4e_pool
const int inception_4e_pool_in_channel = 528;
const int inception_4e_pool_in_height = 14;
const int inception_4e_pool_in_width = 14;
const int inception_4e_pool_pad_top = 1;
const int inception_4e_pool_pad_bottom = 1;
const int inception_4e_pool_pad_left = 1;
const int inception_4e_pool_pad_right = 1;
const int inception_4e_pool_out_channel = inception_4e_pool_in_channel;
const int inception_4e_pool_kernel_height = 3;
const int inception_4e_pool_kernel_width = 3;
const int inception_4e_pool_out_height = 14;
const int inception_4e_pool_out_width = 14;
const int inception_4e_pool_stride = 1;
const int inception_4e_pool_out_channel_DDR_offset = 0;

///layer inception_4e_pool_proj
const int inception_4e_pool_proj_in_channel = 528;
const int inception_4e_pool_proj_in_height = 14;
const int inception_4e_pool_proj_in_width = 14;
const int inception_4e_pool_proj_pad_top = 0;
const int inception_4e_pool_proj_pad_bottom = 0;
const int inception_4e_pool_proj_pad_left = 0;
const int inception_4e_pool_proj_pad_right = 0;
const int inception_4e_pool_proj_kernel_num = 128;
const int inception_4e_pool_proj_kernel_channel = inception_4e_pool_proj_in_channel;
const int inception_4e_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_4e_pool_proj_kernel_height = 1;
const int inception_4e_pool_proj_kernel_width = 1;
const int inception_4e_pool_proj_bias_num = inception_4e_pool_proj_kernel_num;
const int inception_4e_pool_proj_bias_DDR_offset = 0;
const int inception_4e_pool_proj_out_channel = inception_4e_pool_proj_kernel_num;
const int inception_4e_pool_proj_out_height = 14;
const int inception_4e_pool_proj_out_width = 14;
const int inception_4e_pool_proj_stride = 1;
const int inception_4e_pool_proj_out_channel_DDR_offset = 0;

///layer pool4_3x3_s2
const int pool4_3x3_s2_in_channel = 832;
const int pool4_3x3_s2_in_height = 14;
const int pool4_3x3_s2_in_width = 14;
const int pool4_3x3_s2_pad_top = 0;
const int pool4_3x3_s2_pad_bottom = 0;
const int pool4_3x3_s2_pad_left = 2;
const int pool4_3x3_s2_pad_right = 2;
const int pool4_3x3_s2_out_channel = pool4_3x3_s2_in_channel;
const int pool4_3x3_s2_kernel_height = 3;
const int pool4_3x3_s2_kernel_width = 3;
const int pool4_3x3_s2_out_height = 7;
const int pool4_3x3_s2_out_width = 7;
const int pool4_3x3_s2_stride = 2;
const int pool4_3x3_s2_out_channel_DDR_offset = 0;

///layer inception_5a_1x1
const int inception_5a_1x1_in_channel = 832;
const int inception_5a_1x1_in_height = 7;
const int inception_5a_1x1_in_width = 7;
const int inception_5a_1x1_pad_top = 0;
const int inception_5a_1x1_pad_bottom = 0;
const int inception_5a_1x1_pad_left = 0;
const int inception_5a_1x1_pad_right = 0;
const int inception_5a_1x1_kernel_num = 256;
const int inception_5a_1x1_kernel_channel = inception_5a_1x1_in_channel;
const int inception_5a_1x1_kernel_channel_DDR_offset = 0;
const int inception_5a_1x1_kernel_height = 1;
const int inception_5a_1x1_kernel_width = 1;
const int inception_5a_1x1_bias_num = inception_5a_1x1_kernel_num;
const int inception_5a_1x1_bias_DDR_offset = 0;
const int inception_5a_1x1_out_channel = inception_5a_1x1_kernel_num;
const int inception_5a_1x1_out_height = 7;
const int inception_5a_1x1_out_width = 7;
const int inception_5a_1x1_stride = 1;
const int inception_5a_1x1_out_channel_DDR_offset = 0;

///layer inception_5a_3x3_reduce
const int inception_5a_3x3_reduce_in_channel = 832;
const int inception_5a_3x3_reduce_in_height = 7;
const int inception_5a_3x3_reduce_in_width = 7;
const int inception_5a_3x3_reduce_pad_top = 0;
const int inception_5a_3x3_reduce_pad_bottom = 0;
const int inception_5a_3x3_reduce_pad_left = 0;
const int inception_5a_3x3_reduce_pad_right = 0;
const int inception_5a_3x3_reduce_kernel_num = 160;
const int inception_5a_3x3_reduce_kernel_channel = inception_5a_3x3_reduce_in_channel;
const int inception_5a_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_5a_3x3_reduce_kernel_height = 1;
const int inception_5a_3x3_reduce_kernel_width = 1;
const int inception_5a_3x3_reduce_bias_num = inception_5a_3x3_reduce_kernel_num;
const int inception_5a_3x3_reduce_bias_DDR_offset = 0;
const int inception_5a_3x3_reduce_out_channel = inception_5a_3x3_reduce_kernel_num;
const int inception_5a_3x3_reduce_out_height = 7;
const int inception_5a_3x3_reduce_out_width = 7;
const int inception_5a_3x3_reduce_stride = 1;
const int inception_5a_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_5a_3x3
const int inception_5a_3x3_in_channel = 160;
const int inception_5a_3x3_in_height = 7;
const int inception_5a_3x3_in_width = 7;
const int inception_5a_3x3_pad_top = 1;
const int inception_5a_3x3_pad_bottom = 1;
const int inception_5a_3x3_pad_left = 1;
const int inception_5a_3x3_pad_right = 1;
const int inception_5a_3x3_kernel_num = 320;
const int inception_5a_3x3_kernel_channel = inception_5a_3x3_in_channel;
const int inception_5a_3x3_kernel_channel_DDR_offset = 0;
const int inception_5a_3x3_kernel_height = 3;
const int inception_5a_3x3_kernel_width = 3;
const int inception_5a_3x3_bias_num = inception_5a_3x3_kernel_num;
const int inception_5a_3x3_bias_DDR_offset = 0;
const int inception_5a_3x3_out_channel = inception_5a_3x3_kernel_num;
const int inception_5a_3x3_out_height = 7;
const int inception_5a_3x3_out_width = 7;
const int inception_5a_3x3_stride = 1;
const int inception_5a_3x3_out_channel_DDR_offset = 0;

///layer inception_5a_5x5_reduce
const int inception_5a_5x5_reduce_in_channel = 832;
const int inception_5a_5x5_reduce_in_height = 7;
const int inception_5a_5x5_reduce_in_width = 7;
const int inception_5a_5x5_reduce_pad_top = 0;
const int inception_5a_5x5_reduce_pad_bottom = 0;
const int inception_5a_5x5_reduce_pad_left = 0;
const int inception_5a_5x5_reduce_pad_right = 0;
const int inception_5a_5x5_reduce_kernel_num = 32;
const int inception_5a_5x5_reduce_kernel_channel = inception_5a_5x5_reduce_in_channel;
const int inception_5a_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_5a_5x5_reduce_kernel_height = 1;
const int inception_5a_5x5_reduce_kernel_width = 1;
const int inception_5a_5x5_reduce_bias_num = inception_5a_5x5_reduce_kernel_num;
const int inception_5a_5x5_reduce_bias_DDR_offset = 0;
const int inception_5a_5x5_reduce_out_channel = inception_5a_5x5_reduce_kernel_num;
const int inception_5a_5x5_reduce_out_height = 7;
const int inception_5a_5x5_reduce_out_width = 7;
const int inception_5a_5x5_reduce_stride = 1;
const int inception_5a_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_5a_5x5
const int inception_5a_5x5_in_channel = 32;
const int inception_5a_5x5_in_height = 7;
const int inception_5a_5x5_in_width = 7;
const int inception_5a_5x5_pad_top = 2;
const int inception_5a_5x5_pad_bottom = 2;
const int inception_5a_5x5_pad_left = 2;
const int inception_5a_5x5_pad_right = 2;
const int inception_5a_5x5_kernel_num = 128;
const int inception_5a_5x5_kernel_channel = inception_5a_5x5_in_channel;
const int inception_5a_5x5_kernel_channel_DDR_offset = 0;
const int inception_5a_5x5_kernel_height = 5;
const int inception_5a_5x5_kernel_width = 5;
const int inception_5a_5x5_bias_num = inception_5a_5x5_kernel_num;
const int inception_5a_5x5_bias_DDR_offset = 0;
const int inception_5a_5x5_out_channel = inception_5a_5x5_kernel_num;
const int inception_5a_5x5_out_height = 7;
const int inception_5a_5x5_out_width = 7;
const int inception_5a_5x5_stride = 1;
const int inception_5a_5x5_out_channel_DDR_offset = 0;

///layer inception_5a_pool
const int inception_5a_pool_in_channel = 832;
const int inception_5a_pool_in_height = 7;
const int inception_5a_pool_in_width = 7;
const int inception_5a_pool_pad_top = 1;
const int inception_5a_pool_pad_bottom = 1;
const int inception_5a_pool_pad_left = 1;
const int inception_5a_pool_pad_right = 1;
const int inception_5a_pool_out_channel = inception_5a_pool_in_channel;
const int inception_5a_pool_kernel_height = 3;
const int inception_5a_pool_kernel_width = 3;
const int inception_5a_pool_out_height = 7;
const int inception_5a_pool_out_width = 7;
const int inception_5a_pool_stride = 1;
const int inception_5a_pool_out_channel_DDR_offset = 0;

///layer inception_5a_pool_proj
const int inception_5a_pool_proj_in_channel = 832;
const int inception_5a_pool_proj_in_height = 7;
const int inception_5a_pool_proj_in_width = 7;
const int inception_5a_pool_proj_pad_top = 0;
const int inception_5a_pool_proj_pad_bottom = 0;
const int inception_5a_pool_proj_pad_left = 0;
const int inception_5a_pool_proj_pad_right = 0;
const int inception_5a_pool_proj_kernel_num = 128;
const int inception_5a_pool_proj_kernel_channel = inception_5a_pool_proj_in_channel;
const int inception_5a_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_5a_pool_proj_kernel_height = 1;
const int inception_5a_pool_proj_kernel_width = 1;
const int inception_5a_pool_proj_bias_num = inception_5a_pool_proj_kernel_num;
const int inception_5a_pool_proj_bias_DDR_offset = 0;
const int inception_5a_pool_proj_out_channel = inception_5a_pool_proj_kernel_num;
const int inception_5a_pool_proj_out_height = 7;
const int inception_5a_pool_proj_out_width = 7;
const int inception_5a_pool_proj_stride = 1;
const int inception_5a_pool_proj_out_channel_DDR_offset = 0;

///layer inception_5b_1x1
const int inception_5b_1x1_in_channel = 832;
const int inception_5b_1x1_in_height = 7;
const int inception_5b_1x1_in_width = 7;
const int inception_5b_1x1_pad_top = 0;
const int inception_5b_1x1_pad_bottom = 0;
const int inception_5b_1x1_pad_left = 0;
const int inception_5b_1x1_pad_right = 0;
const int inception_5b_1x1_kernel_num = 384;
const int inception_5b_1x1_kernel_channel = inception_5b_1x1_in_channel;
const int inception_5b_1x1_kernel_channel_DDR_offset = 0;
const int inception_5b_1x1_kernel_height = 1;
const int inception_5b_1x1_kernel_width = 1;
const int inception_5b_1x1_bias_num = inception_5b_1x1_kernel_num;
const int inception_5b_1x1_bias_DDR_offset = 0;
const int inception_5b_1x1_out_channel = inception_5b_1x1_kernel_num;
const int inception_5b_1x1_out_height = 7;
const int inception_5b_1x1_out_width = 7;
const int inception_5b_1x1_stride = 1;
const int inception_5b_1x1_out_channel_DDR_offset = 0;

///layer inception_5b_3x3_reduce
const int inception_5b_3x3_reduce_in_channel = 832;
const int inception_5b_3x3_reduce_in_height = 7;
const int inception_5b_3x3_reduce_in_width = 7;
const int inception_5b_3x3_reduce_pad_top = 0;
const int inception_5b_3x3_reduce_pad_bottom = 0;
const int inception_5b_3x3_reduce_pad_left = 0;
const int inception_5b_3x3_reduce_pad_right = 0;
const int inception_5b_3x3_reduce_kernel_num = 192;
const int inception_5b_3x3_reduce_kernel_channel = inception_5b_3x3_reduce_in_channel;
const int inception_5b_3x3_reduce_kernel_channel_DDR_offset = 0;
const int inception_5b_3x3_reduce_kernel_height = 1;
const int inception_5b_3x3_reduce_kernel_width = 1;
const int inception_5b_3x3_reduce_bias_num = inception_5b_3x3_reduce_kernel_num;
const int inception_5b_3x3_reduce_bias_DDR_offset = 0;
const int inception_5b_3x3_reduce_out_channel = inception_5b_3x3_reduce_kernel_num;
const int inception_5b_3x3_reduce_out_height = 7;
const int inception_5b_3x3_reduce_out_width = 7;
const int inception_5b_3x3_reduce_stride = 1;
const int inception_5b_3x3_reduce_out_channel_DDR_offset = 0;

///layer inception_5b_3x3
const int inception_5b_3x3_in_channel = 192;
const int inception_5b_3x3_in_height = 7;
const int inception_5b_3x3_in_width = 7;
const int inception_5b_3x3_pad_top = 1;
const int inception_5b_3x3_pad_bottom = 1;
const int inception_5b_3x3_pad_left = 1;
const int inception_5b_3x3_pad_right = 1;
const int inception_5b_3x3_kernel_num = 384;
const int inception_5b_3x3_kernel_channel = inception_5b_3x3_in_channel;
const int inception_5b_3x3_kernel_channel_DDR_offset = 0;
const int inception_5b_3x3_kernel_height = 3;
const int inception_5b_3x3_kernel_width = 3;
const int inception_5b_3x3_bias_num = inception_5b_3x3_kernel_num;
const int inception_5b_3x3_bias_DDR_offset = 0;
const int inception_5b_3x3_out_channel = inception_5b_3x3_kernel_num;
const int inception_5b_3x3_out_height = 7;
const int inception_5b_3x3_out_width = 7;
const int inception_5b_3x3_stride = 1;
const int inception_5b_3x3_out_channel_DDR_offset = 0;

///layer inception_5b_5x5_reduce
const int inception_5b_5x5_reduce_in_channel = 832;
const int inception_5b_5x5_reduce_in_height = 7;
const int inception_5b_5x5_reduce_in_width = 7;
const int inception_5b_5x5_reduce_pad_top = 0;
const int inception_5b_5x5_reduce_pad_bottom = 0;
const int inception_5b_5x5_reduce_pad_left = 0;
const int inception_5b_5x5_reduce_pad_right = 0;
const int inception_5b_5x5_reduce_kernel_num = 48;
const int inception_5b_5x5_reduce_kernel_channel = inception_5b_5x5_reduce_in_channel;
const int inception_5b_5x5_reduce_kernel_channel_DDR_offset = 0;
const int inception_5b_5x5_reduce_kernel_height = 1;
const int inception_5b_5x5_reduce_kernel_width = 1;
const int inception_5b_5x5_reduce_bias_num = inception_5b_5x5_reduce_kernel_num;
const int inception_5b_5x5_reduce_bias_DDR_offset = 0;
const int inception_5b_5x5_reduce_out_channel = inception_5b_5x5_reduce_kernel_num;
const int inception_5b_5x5_reduce_out_height = 7;
const int inception_5b_5x5_reduce_out_width = 7;
const int inception_5b_5x5_reduce_stride = 1;
const int inception_5b_5x5_reduce_out_channel_DDR_offset = 0;

///layer inception_5b_5x5
const int inception_5b_5x5_in_channel = 48;
const int inception_5b_5x5_in_height = 7;
const int inception_5b_5x5_in_width = 7;
const int inception_5b_5x5_pad_top = 2;
const int inception_5b_5x5_pad_bottom = 2;
const int inception_5b_5x5_pad_left = 2;
const int inception_5b_5x5_pad_right = 2;
const int inception_5b_5x5_kernel_num = 128;
const int inception_5b_5x5_kernel_channel = inception_5b_5x5_in_channel;
const int inception_5b_5x5_kernel_channel_DDR_offset = 0;
const int inception_5b_5x5_kernel_height = 5;
const int inception_5b_5x5_kernel_width = 5;
const int inception_5b_5x5_bias_num = inception_5b_5x5_kernel_num;
const int inception_5b_5x5_bias_DDR_offset = 0;
const int inception_5b_5x5_out_channel = inception_5b_5x5_kernel_num;
const int inception_5b_5x5_out_height = 7;
const int inception_5b_5x5_out_width = 7;
const int inception_5b_5x5_stride = 1;
const int inception_5b_5x5_out_channel_DDR_offset = 0;

///layer inception_5b_pool
const int inception_5b_pool_in_channel = 832;
const int inception_5b_pool_in_height = 7;
const int inception_5b_pool_in_width = 7;
const int inception_5b_pool_pad_top = 1;
const int inception_5b_pool_pad_bottom = 1;
const int inception_5b_pool_pad_left = 1;
const int inception_5b_pool_pad_right = 1;
const int inception_5b_pool_out_channel = inception_5b_pool_in_channel;
const int inception_5b_pool_kernel_height = 3;
const int inception_5b_pool_kernel_width = 3;
const int inception_5b_pool_out_height = 7;
const int inception_5b_pool_out_width = 7;
const int inception_5b_pool_stride = 1;
const int inception_5b_pool_out_channel_DDR_offset = 0;

///layer inception_5b_pool_proj
const int inception_5b_pool_proj_in_channel = 832;
const int inception_5b_pool_proj_in_height = 7;
const int inception_5b_pool_proj_in_width = 7;
const int inception_5b_pool_proj_pad_top = 0;
const int inception_5b_pool_proj_pad_bottom = 0;
const int inception_5b_pool_proj_pad_left = 0;
const int inception_5b_pool_proj_pad_right = 0;
const int inception_5b_pool_proj_kernel_num = 128;
const int inception_5b_pool_proj_kernel_channel = inception_5b_pool_proj_in_channel;
const int inception_5b_pool_proj_kernel_channel_DDR_offset = 0;
const int inception_5b_pool_proj_kernel_height = 1;
const int inception_5b_pool_proj_kernel_width = 1;
const int inception_5b_pool_proj_bias_num = inception_5b_pool_proj_kernel_num;
const int inception_5b_pool_proj_bias_DDR_offset = 0;
const int inception_5b_pool_proj_out_channel = inception_5b_pool_proj_kernel_num;
const int inception_5b_pool_proj_out_height = 7;
const int inception_5b_pool_proj_out_width = 7;
const int inception_5b_pool_proj_stride = 1;
const int inception_5b_pool_proj_out_channel_DDR_offset = 0;

///layer pool5_7x7_s1
const int pool5_7x7_s1_in_channel = 1024;
const int pool5_7x7_s1_in_height = 7;
const int pool5_7x7_s1_in_width = 7;
const int pool5_7x7_s1_pad_top = 0;
const int pool5_7x7_s1_pad_bottom = 0;
const int pool5_7x7_s1_pad_left = 0;
const int pool5_7x7_s1_pad_right = 0;
const int pool5_7x7_s1_out_channel = pool5_7x7_s1_in_channel;
const int pool5_7x7_s1_kernel_height = 7;
const int pool5_7x7_s1_kernel_width = 7;
const int pool5_7x7_s1_out_height = 1;
const int pool5_7x7_s1_out_width = 1;
const int pool5_7x7_s1_stride = 1;
const int pool5_7x7_s1_out_channel_DDR_offset = 0;

/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////


/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////



/////////////////////////////// inception(5a) -> linear              ////////////////////////////


#endif
