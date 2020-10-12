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

//LRN
//avgpool7x7_s1
const int NUM_PE_LRN = 2;
const int N_CHAN_LRN = 4;
const int OUT_HEIGHT_LRN = 1;
const int OUT_WIDTH_LRN = 1;
const int IN_HEIGHT_LRN = OUT_HEIGHT_LRN;
const int IN_WIDTH_LRN = OUT_WIDTH_LRN;

//config of layers
/////header_insert/////
/////////////////////////////// convolution -> inception(3b) max pool////////////////////////////


/////////////////////////////// inception(4a) -> inception(4e) max pool////////////////////////////



/////////////////////////////// inception(5a) -> linear              ////////////////////////////


#endif
