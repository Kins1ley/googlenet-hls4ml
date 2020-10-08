#ifndef TEMPLATE_CONFIG_H_
#define TEMPLATE_CONFIG_H_

//#include "nnet_conv_input_reuse.h"
#include "nnet_conv_output_reuse.h"
#include "nnet_pooling.h"
#include "nnet_relu.h"
#include "nnet_buffer.h"
#include "nnet_lrn.h"
#include "nnet_linear.h"
#include "nnet_common.h"
#include "nnet_bias.h"
#include "header.h"

//this file is the config for PEs reused
//part contained
//---conv2d pe config
//---pooling pe config
//---relu function after conv PEs
//---DRAM weight config
//---global BRAM config
//---local BRAM config
//---template config for layers


//conv2d pe config
struct conv2d_config_1x1_s1 :nnet::conv2d_config
{
	// Internal data type definitions
	typedef FIX_INT32 accum_t;
	typedef FIX_INT8 weight_t;
	typedef FIX_INT20 in_t;
	typedef FIX_INT20 out_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_CONV1x1_S1;
	static const unsigned in_width = IN_WIDTH_CONV1x1_S1;
	static const unsigned n_chan = IN_CHAN_CONV1x1_S1;
	static const unsigned filt_height = 1;
	static const unsigned filt_width = 1;
	static const unsigned n_filt = OUT_CHAN_CONV1x1_S1;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = OUT_HEIGHT_CONV1x1_S1;
	static const unsigned out_width = OUT_WIDTH_CONV1x1_S1;

};


struct conv2d_config_3x3_s1:nnet::conv2d_config
{
	// Internal data type definitions
	typedef FIX_INT32 accum_t;
	typedef FIX_INT8 weight_t;
	typedef FIX_INT20 in_t;
	typedef FIX_INT20 out_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_CONV3x3_S1;
	static const unsigned in_width = IN_WIDTH_CONV3x3_S1;
	static const unsigned n_chan = IN_CHAN_CONV3x3_S1;
	static const unsigned filt_height = 3;
	static const unsigned filt_width = 3;
	static const unsigned n_filt = OUT_CHAN_CONV3x3_S1;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = OUT_HEIGHT_CONV3x3_S1;
	static const unsigned out_width = OUT_WIDTH_CONV3x3_S1;

};

struct conv2d_config_5x5_s1 :nnet::conv2d_config
{
	// Internal data type definitions
	typedef FIX_INT32 accum_t;
	typedef FIX_INT8 weight_t;
	typedef FIX_INT20 in_t;
	typedef FIX_INT20 out_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_CONV5x5_S1;
	static const unsigned in_width = IN_WIDTH_CONV5x5_S1;
	static const unsigned n_chan = IN_CHAN_CONV5x5_S1;
	static const unsigned filt_height = 5;
	static const unsigned filt_width = 5;
	static const unsigned n_filt = OUT_CHAN_CONV5x5_S1;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = OUT_HEIGHT_CONV5x5_S1;
	static const unsigned out_width = OUT_WIDTH_CONV5x5_S1;

};

struct conv2d_config_7x7_s2 :nnet::conv2d_config
{
	// Internal data type definitions
	typedef FIX_INT32 accum_t;
	typedef FIX_INT8 weight_t;
	typedef FIX_INT20 in_t;
	typedef FIX_INT20 out_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_CONV7x7_S2;
	static const unsigned in_width = IN_WIDTH_CONV7x7_S2;
	static const unsigned n_chan = IN_CHAN_CONV7x7_S2;
	static const unsigned filt_height = 7;
	static const unsigned filt_width = 7;
	static const unsigned n_filt = OUT_CHAN_CONV7x7_S2;
	static const unsigned stride_height = 2;
	static const unsigned stride_width = 2;
	static const unsigned out_height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned out_width = OUT_WIDTH_CONV7x7_S2;

};

//pooling pe config
struct pool2d_config_max3x3_s1:nnet::pool2d_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_t;
	typedef FIX_INT32 accum_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_MAXPOOL3x3_S1;
	static const unsigned in_width = IN_WIDTH_MAXPOOL3x3_S1;
	static const unsigned n_chan = N_CHAN_MAXPOOL3x3_S1;
	static const unsigned filt_height = 3;
	static const unsigned filt_width = 3;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = OUT_HEIGHT_MAXPOOL3x3_S1;
	static const unsigned out_width = OUT_WIDTH_MAXPOOL3x3_S1;
	static const unsigned avgpool = 0;

};

struct pool2d_config_max3x3_s2:nnet::pool2d_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_t;
	typedef FIX_INT32 accum_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_MAXPOOL3x3_S2;
	static const unsigned in_width = IN_WIDTH_MAXPOOL3x3_S2;
	static const unsigned n_chan = N_CHAN_MAXPOOL3x3_S2;
	static const unsigned filt_height = 3;
	static const unsigned filt_width = 3;
	static const unsigned stride_height = 2;
	static const unsigned stride_width = 2;
	static const unsigned out_height = OUT_HEIGHT_MAXPOOL3x3_S2;
	static const unsigned out_width = OUT_WIDTH_MAXPOOL3x3_S2;
	static const unsigned avgpool = 0;

};

struct pool2d_config_avg7x7_s1 :nnet::pool2d_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_t;
	typedef FIX_INT32 accum_t;

	// Convolutional parameters
	static const unsigned in_height = IN_HEIGHT_AVGPOOL7x7_S1;
	static const unsigned in_width = IN_WIDTH_AVGPOOL7x7_S1;
	static const unsigned n_chan = N_CHAN_AVGPOOL7x7_S1;
	static const unsigned filt_height = 7;
	static const unsigned filt_width = 7;
	static const unsigned stride_height = 1;
	static const unsigned stride_width = 1;
	static const unsigned out_height = OUT_HEIGHT_AVGPOOL7x7_S1;
	static const unsigned out_width = OUT_WIDTH_AVGPOOL7x7_S1;
	static const unsigned avgpool = 1;

};


//relu function after conv PEs
struct relu_conv2d_config_1x1_s1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;
	static const unsigned in_height = OUT_HEIGHT_CONV1x1_S1;
	static const unsigned in_width = OUT_WIDTH_CONV1x1_S1;
	static const unsigned n_chan = 1;

};


struct relu_conv2d_config_3x3_s1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = OUT_HEIGHT_CONV3x3_S1;
	static const unsigned in_width = OUT_WIDTH_CONV3x3_S1;
	static const unsigned in_channel = 1;

};

struct relu_conv2d_config_5x5_s1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = OUT_HEIGHT_CONV5x5_S1;
	static const unsigned in_width = OUT_WIDTH_CONV5x5_S1;
	static const unsigned in_channel = 1;

};

struct relu_conv2d_config_7x7_s2 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned in_width = OUT_WIDTH_CONV7x7_S2;
	static const unsigned in_channel = 1;

};
//DRAM weight config
struct DDR_weight7x7_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = DDR_WEIGHT_7x7_OUT_CHANNEL;
	static const unsigned in_channel = DDR_WEIGHT_7x7_IN_CHANNEL;
	static const unsigned height = 7;
	static const unsigned width = 7;
};
struct DDR_weight5x5_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = DDR_WEIGHT_5x5_OUT_CHANNEL;
	static const unsigned in_channel = DDR_WEIGHT_5x5_IN_CHANNEL;
	static const unsigned height = 5;
	static const unsigned width = 5;
};
struct DDR_weight3x3_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = DDR_WEIGHT_3x3_OUT_CHANNEL;
	static const unsigned in_channel = DDR_WEIGHT_3x3_IN_CHANNEL;
	static const unsigned height = 3;
	static const unsigned width = 3;
};
struct DDR_weight1x1_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = DDR_WEIGHT_1x1_OUT_CHANNEL;
	static const unsigned in_channel = DDR_WEIGHT_1x1_IN_CHANNEL;
	static const unsigned height = 1;
	static const unsigned width = 1;
};


//global BRAM config
struct global_feature_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;

	static const unsigned channel = CHANNEL_FEATURE_GLOBAL;
	static const unsigned height = HEIGHT_FEATURE_GLOBAL;
	static const unsigned width = WIDTH_FEATURE_GLOBAL;
};

struct WEIGHT_GLOBAL_7x7_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHANNEL_WEIGHT_GLOBAL_7x7;
	static const unsigned in_channel = IN_CHANNEL_WEIGHT_GLOBAL_7x7;
	static const unsigned height = 7;
	static const unsigned width = 7;
};

struct WEIGHT_GLOBAL_5x5_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHANNEL_WEIGHT_GLOBAL_5x5;
	static const unsigned in_channel = IN_CHANNEL_WEIGHT_GLOBAL_5x5;
	static const unsigned height = 5;
	static const unsigned width = 5;
};
struct WEIGHT_GLOBAL_3x3_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHANNEL_WEIGHT_GLOBAL_3x3;
	static const unsigned in_channel = IN_CHANNEL_WEIGHT_GLOBAL_3x3;
	static const unsigned height = 3;
	static const unsigned width = 3;
};

struct WEIGHT_GLOBAL_1x1_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHANNEL_WEIGHT_GLOBAL_1x1;
	static const unsigned in_channel = IN_CHANNEL_WEIGHT_GLOBAL_1x1;
	static const unsigned height = 1;
	static const unsigned width = 1;
};


//local BRAM config
//CONV7x7_S2
struct CONV7x7_S2_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = IN_CHAN_CONV7x7_S2;
	static const unsigned height = IN_HEIGHT_CONV7x7_S2;
	static const unsigned width = IN_WIDTH_CONV7x7_S2;
};

struct CONV7x7_S2_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned in_channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned width = OUT_WIDTH_CONV7x7_S2;
};

struct CONV7x7_S2_local_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHAN_CONV7x7_S2;
	static const unsigned in_channel = IN_CHAN_CONV7x7_S2;
	static const unsigned height = 7;
	static const unsigned width = 7;
};
//CONV5x5_S1
struct CONV5x5_S1_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = IN_CHAN_CONV5x5_S1;
	static const unsigned height = IN_HEIGHT_CONV5x7_S1;
	static const unsigned width = IN_WIDTH_CONV5x5_S1;
};

struct CONV5x5_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned in_channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV5x5_S1;
	static const unsigned width = OUT_WIDTH_CONV5x5_S1;
};

struct CONV5x5_S1_local_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHAN_CONV5x5_S1;
	static const unsigned in_channel = IN_CHAN_CONV5x5_S1;
	static const unsigned height = 5;
	static const unsigned width = 5;
};

//CONV3x3_S1
struct CONV3x3_S1_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = IN_CHAN_CONV3x3_S1;
	static const unsigned height = IN_HEIGHT_CONV5x7_S1;
	static const unsigned width = IN_WIDTH_CONV3x3_S1;
};

struct CONV3x3_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned in_channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV3x3_S1;
	static const unsigned width = OUT_WIDTH_CONV3x3_S1;
};

struct CONV3x3_S1_local_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHAN_CONV3x3_S1;
	static const unsigned in_channel = IN_CHAN_CONV3x3_S1;
	static const unsigned height = 3;
	static const unsigned width = 3;
};
//CONV1x1_S1
struct CONV1x1_S1_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = IN_CHAN_CONV1x1_S1;
	static const unsigned height = IN_HEIGHT_CONV1x1_S1;
	static const unsigned width = IN_WIDTH_CONV1x1_S1;
};

struct CONV1x1_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned in_channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV1x1_S1;
	static const unsigned width = OUT_WIDTH_CONV1x1_S1;
};

struct CONV1x1_S1_local_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHAN_CONV1x1_S1;
	static const unsigned in_channel = IN_CHAN_CONV1x1_S1;
	static const unsigned height = 1;
	static const unsigned width = 1;
};

//MAXPOOL3x3_S2
struct MAXPOOL3x3_S2_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_MAXPOOL3x3_S2;
	static const unsigned height = IN_HEIGHT_MAXPOOL3x3_S2;
	static const unsigned width = IN_WIDTH_MAXPOOL3x3_S2;
};

struct MAXPOOL3x3_S2_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_MAXPOOL3x3_S2;
	static const unsigned height = OUT_HEIGHT_MAXPOOL3x3_S2;
	static const unsigned width = OUT_WIDTH_MAXPOOL3x3_S2;
};
//MAXPOOL3x3_S1
struct MAXPOOL3x3_S1_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_MAXPOOL3x3_S1;
	static const unsigned height = IN_HEIGHT_MAXPOOL3x3_S1;
	static const unsigned width = IN_WIDTH_MAXPOOL3x3_S1;
};

struct MAXPOOL3x3_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_MAXPOOL3x3_S1;
	static const unsigned height = OUT_HEIGHT_MAXPOOL3x3_S1;
	static const unsigned width = OUT_WIDTH_MAXPOOL3x3_S1;
};

//AVGPOOL7x7_S1
struct AVGPOOL7x7_S1_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_AVGPOOL7x7_S1;
	static const unsigned height = IN_HEIGHT_AVGPOOL7x7_S1;
	static const unsigned width = IN_WIDTH_AVGPOOL7x7_S1;
};

struct AVGPOOL7x7_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_AVGPOOL7x7_S1;
	static const unsigned height = OUT_HEIGHT_AVGPOOL7x7_S1;
	static const unsigned width = OUT_WIDTH_AVGPOOL7x7_S1;
};
//LRN
struct LRN_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_LRN;
	static const unsigned height = IN_HEIGHT_LRN;
	static const unsigned width = IN_WIDTH_LRN;
};

struct LRN_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = N_CHAN_LRN;
	static const unsigned height = OUT_HEIGHT_LRN;
	static const unsigned width = OUT_WIDTH_LRN;
};

//template config for layers
struct DDR_feature_data_0_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = IMAGE_CH;
	static const unsigned height = IMAGE_H;
	static const unsigned width = IMAGE_W;
};
/////template_config_insert/////
///conv1_7x7_s2
struct DDR_feature_conv1_7x7_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = conv1_7x7_s2_out_channel;
	static const unsigned height = conv1_7x7_s2_out_height;
	static const unsigned width = conv1_7x7_s2_out_width;
};
///pool1_3x3_s2
struct DDR_feature_pool1_3x3_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = pool1_3x3_s2_out_channel;
	static const unsigned height = pool1_3x3_s2_out_height;
	static const unsigned width = pool1_3x3_s2_out_width;
};
///pool1_norm1
struct DDR_feature_pool1_norm1_1_config : nnet::Feature_Memory {
    typedef FIX_INT20 feature_type;
    static const unsigned channel = pool1_norm1_out_channel;
    static const unsigned height = pool1_norm1_out_height;
    static const unsigned width = pool1_norm1_out_width;
};
///conv2_3x3_reduce
struct DDR_feature_conv2_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = conv2_3x3_reduce_out_channel;
	static const unsigned height = conv2_3x3_reduce_out_height;
	static const unsigned width = conv2_3x3_reduce_out_width;
};
///conv2_3x3
struct DDR_feature_conv2_3x3_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = conv2_3x3_out_channel;
	static const unsigned height = conv2_3x3_out_height;
	static const unsigned width = conv2_3x3_out_width;
};
///conv2_norm2
struct DDR_feature_conv2_norm2_1_config : nnet::Feature_Memory {
    typedef FIX_INT20 feature_type;
    static const unsigned channel = conv2_norm2_out_channel;
    static const unsigned height = conv2_norm2_out_height;
    static const unsigned width = conv2_norm2_out_width;
};
///pool2_3x3_s2
struct DDR_feature_pool2_3x3_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = pool2_3x3_s2_out_channel;
	static const unsigned height = pool2_3x3_s2_out_height;
	static const unsigned width = pool2_3x3_s2_out_width;
};
///inception_3a_1x1
struct DDR_feature_inception_3a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_1x1_out_channel;
	static const unsigned height = inception_3a_1x1_out_height;
	static const unsigned width = inception_3a_1x1_out_width;
};
///inception_3a_3x3_reduce
struct DDR_feature_inception_3a_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_3x3_reduce_out_channel;
	static const unsigned height = inception_3a_3x3_reduce_out_height;
	static const unsigned width = inception_3a_3x3_reduce_out_width;
};
///inception_3a_3x3
struct DDR_feature_inception_3a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_3x3_out_channel;
	static const unsigned height = inception_3a_3x3_out_height;
	static const unsigned width = inception_3a_3x3_out_width;
};
///inception_3a_5x5_reduce
struct DDR_feature_inception_3a_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_5x5_reduce_out_channel;
	static const unsigned height = inception_3a_5x5_reduce_out_height;
	static const unsigned width = inception_3a_5x5_reduce_out_width;
};
///inception_3a_5x5
struct DDR_feature_inception_3a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_5x5_out_channel;
	static const unsigned height = inception_3a_5x5_out_height;
	static const unsigned width = inception_3a_5x5_out_width;
};
///inception_3a_pool
struct DDR_feature_inception_3a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_pool_out_channel;
	static const unsigned height = inception_3a_pool_out_height;
	static const unsigned width = inception_3a_pool_out_width;
};
///inception_3a_pool_proj
struct DDR_feature_inception_3a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3a_pool_proj_out_channel;
	static const unsigned height = inception_3a_pool_proj_out_height;
	static const unsigned width = inception_3a_pool_proj_out_width;
};
///inception_3b_1x1
struct DDR_feature_inception_3b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_1x1_out_channel;
	static const unsigned height = inception_3b_1x1_out_height;
	static const unsigned width = inception_3b_1x1_out_width;
};
///inception_3b_3x3_reduce
struct DDR_feature_inception_3b_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_3x3_reduce_out_channel;
	static const unsigned height = inception_3b_3x3_reduce_out_height;
	static const unsigned width = inception_3b_3x3_reduce_out_width;
};
///inception_3b_3x3
struct DDR_feature_inception_3b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_3x3_out_channel;
	static const unsigned height = inception_3b_3x3_out_height;
	static const unsigned width = inception_3b_3x3_out_width;
};
///inception_3b_5x5_reduce
struct DDR_feature_inception_3b_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_5x5_reduce_out_channel;
	static const unsigned height = inception_3b_5x5_reduce_out_height;
	static const unsigned width = inception_3b_5x5_reduce_out_width;
};
///inception_3b_5x5
struct DDR_feature_inception_3b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_5x5_out_channel;
	static const unsigned height = inception_3b_5x5_out_height;
	static const unsigned width = inception_3b_5x5_out_width;
};
///inception_3b_pool
struct DDR_feature_inception_3b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_pool_out_channel;
	static const unsigned height = inception_3b_pool_out_height;
	static const unsigned width = inception_3b_pool_out_width;
};
///inception_3b_pool_proj
struct DDR_feature_inception_3b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_3b_pool_proj_out_channel;
	static const unsigned height = inception_3b_pool_proj_out_height;
	static const unsigned width = inception_3b_pool_proj_out_width;
};
///pool3_3x3_s2
struct DDR_feature_pool3_3x3_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = pool3_3x3_s2_out_channel;
	static const unsigned height = pool3_3x3_s2_out_height;
	static const unsigned width = pool3_3x3_s2_out_width;
};
///inception_4a_1x1
struct DDR_feature_inception_4a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_1x1_out_channel;
	static const unsigned height = inception_4a_1x1_out_height;
	static const unsigned width = inception_4a_1x1_out_width;
};
///inception_4a_3x3_reduce
struct DDR_feature_inception_4a_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_3x3_reduce_out_channel;
	static const unsigned height = inception_4a_3x3_reduce_out_height;
	static const unsigned width = inception_4a_3x3_reduce_out_width;
};
///inception_4a_3x3
struct DDR_feature_inception_4a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_3x3_out_channel;
	static const unsigned height = inception_4a_3x3_out_height;
	static const unsigned width = inception_4a_3x3_out_width;
};
///inception_4a_5x5_reduce
struct DDR_feature_inception_4a_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_5x5_reduce_out_channel;
	static const unsigned height = inception_4a_5x5_reduce_out_height;
	static const unsigned width = inception_4a_5x5_reduce_out_width;
};
///inception_4a_5x5
struct DDR_feature_inception_4a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_5x5_out_channel;
	static const unsigned height = inception_4a_5x5_out_height;
	static const unsigned width = inception_4a_5x5_out_width;
};
///inception_4a_pool
struct DDR_feature_inception_4a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_pool_out_channel;
	static const unsigned height = inception_4a_pool_out_height;
	static const unsigned width = inception_4a_pool_out_width;
};
///inception_4a_pool_proj
struct DDR_feature_inception_4a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4a_pool_proj_out_channel;
	static const unsigned height = inception_4a_pool_proj_out_height;
	static const unsigned width = inception_4a_pool_proj_out_width;
};
///inception_4b_1x1
struct DDR_feature_inception_4b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_1x1_out_channel;
	static const unsigned height = inception_4b_1x1_out_height;
	static const unsigned width = inception_4b_1x1_out_width;
};
///inception_4b_3x3_reduce
struct DDR_feature_inception_4b_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_3x3_reduce_out_channel;
	static const unsigned height = inception_4b_3x3_reduce_out_height;
	static const unsigned width = inception_4b_3x3_reduce_out_width;
};
///inception_4b_3x3
struct DDR_feature_inception_4b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_3x3_out_channel;
	static const unsigned height = inception_4b_3x3_out_height;
	static const unsigned width = inception_4b_3x3_out_width;
};
///inception_4b_5x5_reduce
struct DDR_feature_inception_4b_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_5x5_reduce_out_channel;
	static const unsigned height = inception_4b_5x5_reduce_out_height;
	static const unsigned width = inception_4b_5x5_reduce_out_width;
};
///inception_4b_5x5
struct DDR_feature_inception_4b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_5x5_out_channel;
	static const unsigned height = inception_4b_5x5_out_height;
	static const unsigned width = inception_4b_5x5_out_width;
};
///inception_4b_pool
struct DDR_feature_inception_4b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_pool_out_channel;
	static const unsigned height = inception_4b_pool_out_height;
	static const unsigned width = inception_4b_pool_out_width;
};
///inception_4b_pool_proj
struct DDR_feature_inception_4b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4b_pool_proj_out_channel;
	static const unsigned height = inception_4b_pool_proj_out_height;
	static const unsigned width = inception_4b_pool_proj_out_width;
};
///inception_4c_1x1
struct DDR_feature_inception_4c_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_1x1_out_channel;
	static const unsigned height = inception_4c_1x1_out_height;
	static const unsigned width = inception_4c_1x1_out_width;
};
///inception_4c_3x3_reduce
struct DDR_feature_inception_4c_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_3x3_reduce_out_channel;
	static const unsigned height = inception_4c_3x3_reduce_out_height;
	static const unsigned width = inception_4c_3x3_reduce_out_width;
};
///inception_4c_3x3
struct DDR_feature_inception_4c_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_3x3_out_channel;
	static const unsigned height = inception_4c_3x3_out_height;
	static const unsigned width = inception_4c_3x3_out_width;
};
///inception_4c_5x5_reduce
struct DDR_feature_inception_4c_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_5x5_reduce_out_channel;
	static const unsigned height = inception_4c_5x5_reduce_out_height;
	static const unsigned width = inception_4c_5x5_reduce_out_width;
};
///inception_4c_5x5
struct DDR_feature_inception_4c_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_5x5_out_channel;
	static const unsigned height = inception_4c_5x5_out_height;
	static const unsigned width = inception_4c_5x5_out_width;
};
///inception_4c_pool
struct DDR_feature_inception_4c_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_pool_out_channel;
	static const unsigned height = inception_4c_pool_out_height;
	static const unsigned width = inception_4c_pool_out_width;
};
///inception_4c_pool_proj
struct DDR_feature_inception_4c_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4c_pool_proj_out_channel;
	static const unsigned height = inception_4c_pool_proj_out_height;
	static const unsigned width = inception_4c_pool_proj_out_width;
};
///inception_4d_1x1
struct DDR_feature_inception_4d_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_1x1_out_channel;
	static const unsigned height = inception_4d_1x1_out_height;
	static const unsigned width = inception_4d_1x1_out_width;
};
///inception_4d_3x3_reduce
struct DDR_feature_inception_4d_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_3x3_reduce_out_channel;
	static const unsigned height = inception_4d_3x3_reduce_out_height;
	static const unsigned width = inception_4d_3x3_reduce_out_width;
};
///inception_4d_3x3
struct DDR_feature_inception_4d_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_3x3_out_channel;
	static const unsigned height = inception_4d_3x3_out_height;
	static const unsigned width = inception_4d_3x3_out_width;
};
///inception_4d_5x5_reduce
struct DDR_feature_inception_4d_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_5x5_reduce_out_channel;
	static const unsigned height = inception_4d_5x5_reduce_out_height;
	static const unsigned width = inception_4d_5x5_reduce_out_width;
};
///inception_4d_5x5
struct DDR_feature_inception_4d_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_5x5_out_channel;
	static const unsigned height = inception_4d_5x5_out_height;
	static const unsigned width = inception_4d_5x5_out_width;
};
///inception_4d_pool
struct DDR_feature_inception_4d_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_pool_out_channel;
	static const unsigned height = inception_4d_pool_out_height;
	static const unsigned width = inception_4d_pool_out_width;
};
///inception_4d_pool_proj
struct DDR_feature_inception_4d_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4d_pool_proj_out_channel;
	static const unsigned height = inception_4d_pool_proj_out_height;
	static const unsigned width = inception_4d_pool_proj_out_width;
};
///inception_4e_1x1
struct DDR_feature_inception_4e_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_1x1_out_channel;
	static const unsigned height = inception_4e_1x1_out_height;
	static const unsigned width = inception_4e_1x1_out_width;
};
///inception_4e_3x3_reduce
struct DDR_feature_inception_4e_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_3x3_reduce_out_channel;
	static const unsigned height = inception_4e_3x3_reduce_out_height;
	static const unsigned width = inception_4e_3x3_reduce_out_width;
};
///inception_4e_3x3
struct DDR_feature_inception_4e_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_3x3_out_channel;
	static const unsigned height = inception_4e_3x3_out_height;
	static const unsigned width = inception_4e_3x3_out_width;
};
///inception_4e_5x5_reduce
struct DDR_feature_inception_4e_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_5x5_reduce_out_channel;
	static const unsigned height = inception_4e_5x5_reduce_out_height;
	static const unsigned width = inception_4e_5x5_reduce_out_width;
};
///inception_4e_5x5
struct DDR_feature_inception_4e_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_5x5_out_channel;
	static const unsigned height = inception_4e_5x5_out_height;
	static const unsigned width = inception_4e_5x5_out_width;
};
///inception_4e_pool
struct DDR_feature_inception_4e_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_pool_out_channel;
	static const unsigned height = inception_4e_pool_out_height;
	static const unsigned width = inception_4e_pool_out_width;
};
///inception_4e_pool_proj
struct DDR_feature_inception_4e_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_4e_pool_proj_out_channel;
	static const unsigned height = inception_4e_pool_proj_out_height;
	static const unsigned width = inception_4e_pool_proj_out_width;
};
///pool4_3x3_s2
struct DDR_feature_pool4_3x3_s2_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = pool4_3x3_s2_out_channel;
	static const unsigned height = pool4_3x3_s2_out_height;
	static const unsigned width = pool4_3x3_s2_out_width;
};
///inception_5a_1x1
struct DDR_feature_inception_5a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_1x1_out_channel;
	static const unsigned height = inception_5a_1x1_out_height;
	static const unsigned width = inception_5a_1x1_out_width;
};
///inception_5a_3x3_reduce
struct DDR_feature_inception_5a_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_3x3_reduce_out_channel;
	static const unsigned height = inception_5a_3x3_reduce_out_height;
	static const unsigned width = inception_5a_3x3_reduce_out_width;
};
///inception_5a_3x3
struct DDR_feature_inception_5a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_3x3_out_channel;
	static const unsigned height = inception_5a_3x3_out_height;
	static const unsigned width = inception_5a_3x3_out_width;
};
///inception_5a_5x5_reduce
struct DDR_feature_inception_5a_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_5x5_reduce_out_channel;
	static const unsigned height = inception_5a_5x5_reduce_out_height;
	static const unsigned width = inception_5a_5x5_reduce_out_width;
};
///inception_5a_5x5
struct DDR_feature_inception_5a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_5x5_out_channel;
	static const unsigned height = inception_5a_5x5_out_height;
	static const unsigned width = inception_5a_5x5_out_width;
};
///inception_5a_pool
struct DDR_feature_inception_5a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_pool_out_channel;
	static const unsigned height = inception_5a_pool_out_height;
	static const unsigned width = inception_5a_pool_out_width;
};
///inception_5a_pool_proj
struct DDR_feature_inception_5a_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5a_pool_proj_out_channel;
	static const unsigned height = inception_5a_pool_proj_out_height;
	static const unsigned width = inception_5a_pool_proj_out_width;
};
///inception_5b_1x1
struct DDR_feature_inception_5b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_1x1_out_channel;
	static const unsigned height = inception_5b_1x1_out_height;
	static const unsigned width = inception_5b_1x1_out_width;
};
///inception_5b_3x3_reduce
struct DDR_feature_inception_5b_3x3_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_3x3_reduce_out_channel;
	static const unsigned height = inception_5b_3x3_reduce_out_height;
	static const unsigned width = inception_5b_3x3_reduce_out_width;
};
///inception_5b_3x3
struct DDR_feature_inception_5b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_3x3_out_channel;
	static const unsigned height = inception_5b_3x3_out_height;
	static const unsigned width = inception_5b_3x3_out_width;
};
///inception_5b_5x5_reduce
struct DDR_feature_inception_5b_5x5_reduce_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_5x5_reduce_out_channel;
	static const unsigned height = inception_5b_5x5_reduce_out_height;
	static const unsigned width = inception_5b_5x5_reduce_out_width;
};
///inception_5b_5x5
struct DDR_feature_inception_5b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_5x5_out_channel;
	static const unsigned height = inception_5b_5x5_out_height;
	static const unsigned width = inception_5b_5x5_out_width;
};
///inception_5b_pool
struct DDR_feature_inception_5b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_pool_out_channel;
	static const unsigned height = inception_5b_pool_out_height;
	static const unsigned width = inception_5b_pool_out_width;
};
///inception_5b_pool_proj
struct DDR_feature_inception_5b_output_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = inception_5b_pool_proj_out_channel;
	static const unsigned height = inception_5b_pool_proj_out_height;
	static const unsigned width = inception_5b_pool_proj_out_width;
};
///pool5_7x7_s1
struct DDR_feature_pool5_7x7_s1_1_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = pool5_7x7_s1_out_channel;
	static const unsigned height = pool5_7x7_s1_out_height;
	static const unsigned width = pool5_7x7_s1_out_width;
};
/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)


/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////(Binwu)



/////////////////////////////// inception(5a) -> linear                ////////////////////////////(Qi)



#endif