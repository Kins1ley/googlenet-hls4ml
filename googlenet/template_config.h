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
//conv2d output_reuse 
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

//pooling 
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
	static const unsigned in_height = IN_HEIGHT_CONV1x1_S1;
	static const unsigned in_width = IN_WIDTH_CONV1x1_S1;
	static const unsigned n_chan = 1;

};


struct relu_conv2d_config_3x3_s1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = IN_HEIGHT_CONV3x3_S1;
	static const unsigned in_width = IN_WIDTH_CONV3x3_S1;
	static const unsigned in_channel = 1;

};

struct relu_conv2d_config_5x5_s1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = IN_HEIGHT_CONV5x5_S1;
	static const unsigned in_width = IN_WIDTH_CONV5x5_S1;
	static const unsigned in_channel = 1;

};

struct relu_conv2d_config_7x7_s2 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = IN_HEIGHT_CONV7x7_S2;
	static const unsigned in_width = IN_WIDTH_CONV7x7_S2;
	static const unsigned in_channel = 1;

};


//template config for layers

/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)
///conv1_7x7_s2
struct DDR_feature_image_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = conv1_7x7_s2_in_channel;
	static const unsigned height = conv1_7x7_s2_in_height;
	static const unsigned width = conv1_7x7_s2_in_width;
};
struct global_feature_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;

	static const unsigned channel = CHANNEL_FEATURE_GLOBAL;
	static const unsigned height = HEIGHT_FEATURE_GLOBAL;
	static const unsigned width = WIDTH_FEATURE_GLOBAL;
};
struct conv7x7_s2_local_feature_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned in_channel = 1;
	static const unsigned height = IN_HEIGHT_CONV7x7_S2;
	static const unsigned width = IN_WIDTH_CONV7x7_S2;
};
struct conv7x7_s2_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned in_channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned width = OUT_WIDTH_CONV7x7_S2;
};

struct conv7x7_DDR_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = 64;
	static const unsigned in_channel = 3;
	static const unsigned height = 7;
	static const unsigned width = 7;
};
struct conv7x7_global_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHANNEL_WEIGHT_GLOBAL_7x7;
	static const unsigned in_channel = IN_CHANNEL_WEIGHT_GLOBAL_7x7;
	static const unsigned height = 7;
	static const unsigned width = 7;
};
struct conv7x7_s2_local_weight_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = OUT_CHAN_CONV7x7_S2;
	static const unsigned in_channel = IN_CHAN_CONV7x7_S2;
	static const unsigned height = 7;
	static const unsigned width = 7;
};

struct conv7x7_s2_set_bias_config : nnet::set_bias_config {
	typedef FIX_INT20 bias_type;
	typedef FIX_INT20 feature_type;

	static const unsigned channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned width = OUT_WIDTH_CONV7x7_S2;
};
/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////(Binwu)



/////////////////////////////// inception(5a) -> linear                ////////////////////////////(Qi)



#endif