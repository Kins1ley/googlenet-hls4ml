#ifndef HEADER_H_
#define HEADER_H_

#include <cstdlib>


#include <iostream>
typedef int FIX_INT8;
typedef int FIX_INT20;
typedef int FIX_INT32;

#define DIV_CEIL(x,y) ((x)/(y)+(((x)%(y)==0)?0:1))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

const int IMAGE_CH = 3; // image channel
const int IMAGE_H = 224;// image height
const int IMAGE_W = 224;// image width
const int OU = 1000;//number of final output
///config of global BRAM
const int NUM_WEIGHT_GLOBAL_7x7 = 8 ; //number of global bram of weights
const int OUT_CHANNEL_WEIGHT_GLOBAL_7x7 = 2;//size of global bram of weights
const int IN_CHANNEL_WEIGHT_GLOBAL_7x7 = 64;//size of global bram of weights
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
const int CHANNEL_FEATURE_GLOBAL = 8;//size of global bram of feature
const int WIDTH_FEATURE_GLOBAL = 22;//size of global bram of feature
const int HEIGHT_FEATURE_GLOBAL = 22;//size of global bram of feature

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
const int IN_CHAN_CONV7x7_S2 = 2;
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
const int N_CHAN_MAXPOOL3x3_S2 = 2;
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
/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////
///layer conv1_7x7_s2
/////header_insert_1/////
const int conv1_7x7_s2_in_channel = IMAGE_CH;
const int conv1_7x7_s2_in_height = IMAGE_H;
const int conv1_7x7_s2_in_width = IMAGE_W;
const int conv1_7x7_s2_pad_top = 3;
const int conv1_7x7_s2_pad_bottom = 3;
const int conv1_7x7_s2_pad_left = 3;
const int conv1_7x7_s2_pad_right = 3;
const int conv1_7x7_s2_kernel_num = 528;
const int conv1_7x7_s2_kernel_channel = conv1_7x7_s2_in_channel;
const int conv1_7x7_s2_kernel_height = 7;
const int conv1_7x7_s2_kernel_width = 7;
const int conv1_7x7_s2_out_channel = conv1_7x7_s2_kernel_num;
const int conv1_7x7_s2_out_height = 14;
const int conv1_7x7_s2_out_width = 14;
///layer pool1_3x3_s2
const int pool1_3x3_s2_in_channel = 528;
const int pool1_3x3_s2_in_height = 14;
const int pool1_3x3_s2_in_width = 14;
const int pool1_3x3_s2_pad_top = 1;
const int pool1_3x3_s2_pad_bottom = 1;
const int pool1_3x3_s2_pad_left = 1;
const int pool1_3x3_s2_pad_right = 1;
const int pool1_3x3_s2_out_channel = pool1_3x3_s2_in_channel;
const int pool1_3x3_s2_kernel_height = 3;
const int pool1_3x3_s2_kernel_width = 3;
const int pool1_3x3_s2_out_height = 14;
const int pool1_3x3_s2_out_width = 14;
const int pool1_3x3_s2_stride = 1;
const int pool1_3x3_s2_out_channel_DDR_offset = 0;
/*
const int conv1_7x7_s2_in_channel = IMAGE_CH;
const int conv1_7x7_s2_in_height = IMAGE_H;
const int conv1_7x7_s2_in_width = IMAGE_W;
const int conv1_7x7_s2_pad_top = 3;
const int conv1_7x7_s2_pad_bottom = 3;
const int conv1_7x7_s2_pad_left = 3;
const int conv1_7x7_s2_pad_right = 3;
const int conv1_7x7_s2_kernel_num = 4;
const int conv1_7x7_s2_kernel_channel = conv1_7x7_s2_in_channel;
const int conv1_7x7_s2_kernel_height = 7;
const int conv1_7x7_s2_kernel_width = 7;
const int conv1_7x7_s2_bias_num = conv1_7x7_s2_kernel_num;
const int conv1_7x7_s2_out_channel = conv1_7x7_s2_kernel_num;


const int conv1_7x7_s2_out_height = 56;
const int conv1_7x7_s2_out_width = 56;

///layer pool1_3x3_s2
const int pool1_3x3_s2_in_channel = 4;
const int pool1_3x3_s2_in_height = 56;
const int pool1_3x3_s2_in_width = 56;
const int pool1_3x3_s2_pad_top = 0;
const int pool1_3x3_s2_pad_bottom = 2;
const int pool1_3x3_s2_pad_left = 0;
const int pool1_3x3_s2_pad_right = 2;
const int pool1_3x3_s2_out_channel = pool1_3x3_s2_in_channel;
const int pool1_3x3_s2_kernel_height = 3;
const int pool1_3x3_s2_kernel_width = 3;
const int pool1_3x3_s2_out_height = 28;
const int pool1_3x3_s2_out_width = 28;
const int pool1_3x3_s2_stride = 2;
const int pool1_3x3_s2_out_channel_DDR_offset = 0;*/

///layer conv2_3x3_reduce
const int conv2_3x3_reduce_in_channel = 3;
const int conv2_3x3_reduce_in_height = 224;
const int conv2_3x3_reduce_in_width = 224;
const int conv2_3x3_reduce_pad_top = 1;
const int conv2_3x3_reduce_pad_bottom = 1;
const int conv2_3x3_reduce_pad_left = 1;
const int conv2_3x3_reduce_pad_right = 1;
const int conv2_3x3_reduce_kernel_num = 64;
const int conv2_3x3_reduce_kernel_channel = conv2_3x3_reduce_in_channel;
const int conv2_3x3_reduce_kernel_height = 7;
const int conv2_3x3_reduce_kernel_width = 7;
const int conv2_3x3_reduce_bias_num = conv2_3x3_reduce_kernel_num;
const int conv2_3x3_reduce_out_channel = conv2_3x3_reduce_kernel_num;
const int conv2_3x3_reduce_out_height = 112;
const int conv2_3x3_reduce_out_width = 112;

///layer conv2_3x3
const int conv2_3x3_in_channel = 3;
const int conv2_3x3_in_height = 224;
const int conv2_3x3_in_width = 224;
const int conv2_3x3_pad_top = 1;
const int conv2_3x3_pad_bottom = 1;
const int conv2_3x3_pad_left = 1;
const int conv2_3x3_pad_right = 1;
const int conv2_3x3_kernel_num = 64;
const int conv2_3x3_kernel_channel = conv2_3x3_in_channel;
const int conv2_3x3_kernel_height = 7;
const int conv2_3x3_kernel_width = 7;
const int conv2_3x3_bias_num = conv2_3x3_kernel_num;
const int conv2_3x3_out_channel = conv2_3x3_kernel_num;
const int conv2_3x3_out_height = 112;
const int conv2_3x3_out_width = 112;

///layer pool2_3x3_s2
const int pool2_3x3_s2_in_channel = 832;
const int pool2_3x3_s2_in_height = 14;
const int pool2_3x3_s2_in_width = 14;
const int pool2_3x3_s2_pad_top = 0;
const int pool2_3x3_s2_pad_bottom = 2;
const int pool2_3x3_s2_pad_left = 0;
const int pool2_3x3_s2_pad_right = 2;
const int pool2_3x3_s2_out_channel = pool2_3x3_s2_in_channel;
const int pool2_3x3_s2_out_height = 7;
const int pool2_3x3_s2_out_width = 7;



///layer pool3_3x3_s2
const int pool3_3x3_s2_in_channel = 832;
const int pool3_3x3_s2_in_height = 14;
const int pool3_3x3_s2_in_width = 14;
const int pool3_3x3_s2_pad_top = 0;
const int pool3_3x3_s2_pad_bottom = 2;
const int pool3_3x3_s2_pad_left = 0;
const int pool3_3x3_s2_pad_right = 2;
const int pool3_3x3_s2_out_channel = pool3_3x3_s2_in_channel;
const int pool3_3x3_s2_out_height = 7;
const int pool3_3x3_s2_out_width = 7;


/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////
///layer pool4_3x3_s2
const int pool4_3x3_s2_in_channel = 832;
const int pool4_3x3_s2_in_height = 14;
const int pool4_3x3_s2_in_width = 14;
const int pool4_3x3_s2_pad_top = 0;
const int pool4_3x3_s2_pad_bottom = 2;
const int pool4_3x3_s2_pad_left = 0;
const int pool4_3x3_s2_pad_right = 2;
const int pool4_3x3_s2_out_channel = pool4_3x3_s2_in_channel;
const int pool4_3x3_s2_out_height = 7;
const int pool4_3x3_s2_out_width = 7;




/////////////////////////////// inception(5a) -> linear              ////////////////////////////


#endif
