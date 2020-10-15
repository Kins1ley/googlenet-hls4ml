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
//conv2d pe config
struct conv2d_config_CONV1x1_S1 :nnet::conv2d_config
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
struct CONV1x1_S1_set_bias_config : nnet::set_bias_config {
	typedef FIX_INT20 bias_type;
	typedef FIX_INT20 feature_type;

	static const unsigned channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV1x1_S1;
	static const unsigned width = OUT_WIDTH_CONV1x1_S1;
};

struct conv2d_config_CONV3x3_S1 :nnet::conv2d_config
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
struct CONV3x3_S1_set_bias_config : nnet::set_bias_config {
	typedef FIX_INT20 bias_type;
	typedef FIX_INT20 feature_type;

	static const unsigned channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV3x3_S1;
	static const unsigned width = OUT_WIDTH_CONV3x3_S1;
};

struct conv2d_config_CONV5x5_S1 :nnet::conv2d_config
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
struct CONV5x5_S1_set_bias_config : nnet::set_bias_config {
	typedef FIX_INT20 bias_type;
	typedef FIX_INT20 feature_type;

	static const unsigned channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV5x5_S1;
	static const unsigned width = OUT_WIDTH_CONV5x5_S1;
};

struct conv2d_config_CONV7x7_S2 :nnet::conv2d_config
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
struct CONV7x7_S2_set_bias_config : nnet::set_bias_config {
	typedef FIX_INT20 bias_type;
	typedef FIX_INT20 feature_type;

	static const unsigned channel = 1;
	static const unsigned height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned width = OUT_WIDTH_CONV7x7_S2;
};
//pooling 
struct pool2d_config_MAX3x3_S1:nnet::pool2d_config
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

struct pool2d_config_MAX3x3_S2:nnet::pool2d_config
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

struct pool2d_config_AVG7x7_S1 :nnet::pool2d_config
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
struct relu_conv2d_config_CONV1x1_S1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;
	static const unsigned in_height = OUT_HEIGHT_CONV1x1_S1;
	static const unsigned in_width = OUT_WIDTH_CONV1x1_S1;
	static const unsigned n_chan = 1;

};


struct relu_conv2d_config_CONV3x3_S1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = OUT_HEIGHT_CONV3x3_S1;
	static const unsigned in_width = OUT_WIDTH_CONV3x3_S1;
	static const unsigned in_channel = 1;

};

struct relu_conv2d_config_CONV5x5_S1 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = OUT_HEIGHT_CONV5x5_S1;
	static const unsigned in_width = OUT_WIDTH_CONV5x5_S1;
	static const unsigned in_channel = 1;

};

struct relu_conv2d_config_CONV7x7_S2 :nnet::relu_config
{
	// Internal data type definitions
	typedef FIX_INT20 feature_type;

	static const unsigned in_height = OUT_HEIGHT_CONV7x7_S2;
	static const unsigned in_width = OUT_WIDTH_CONV7x7_S2;
	static const unsigned in_channel = 1;

};
//DRAM config
struct DDR_weight_7x7_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = conv1_7x7_s2_out_channel;
	static const unsigned in_channel = conv1_7x7_s2_in_channel;
	static const unsigned height = conv1_7x7_s2_kernel_height;
	static const unsigned width = conv1_7x7_s2_kernel_width;
};
struct DDR_weight_5x5_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = 1024;
	static const unsigned in_channel = 1024;
	static const unsigned height = 5;
	static const unsigned width = 5;
};
struct DDR_weight_3x3_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = 1024;
	static const unsigned in_channel = 1024;
	static const unsigned height = 3;
	static const unsigned width = 3;
};
struct DDR_weight_1x1_config : nnet::Weight_Memory {
	typedef FIX_INT20 weight_type;
	static const unsigned out_channel = 1024;
	static const unsigned in_channel = 1024;
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
	static const unsigned channel = 1;
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
	static const unsigned height = IN_HEIGHT_CONV5x5_S1;
	static const unsigned width = IN_WIDTH_CONV5x5_S1;
};

struct CONV5x5_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = 1;
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
	static const unsigned height = IN_HEIGHT_CONV3x3_S1;
	static const unsigned width = IN_WIDTH_CONV3x3_S1;
};

struct CONV3x3_S1_local_feature_out_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = 1;
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
	static const unsigned channel = 1;
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
//template config for layers

/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////(Junpeng)
///conv1_7x7_s2
struct DDR_feature_image_in_config : nnet::Feature_Memory {
	typedef FIX_INT20 feature_type;
	static const unsigned channel = conv1_7x7_s2_in_channel;
	static const unsigned height = conv1_7x7_s2_in_height;
	static const unsigned width = conv1_7x7_s2_in_width;
};


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




/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////(Binwu)



/////////////////////////////// inception(5a) -> linear                ////////////////////////////(Qi)



#endif