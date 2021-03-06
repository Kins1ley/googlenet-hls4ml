#ifndef SCHEDULE_CONFIG_H_
#define SCHEDULE_CONFIG_H_

#include "header.h"
//conv1_7x7_s2
const int conv1_7x7_s2_out_height_per_block = ((HEIGHT_FEATURE_GLOBAL - conv1_7x7_s2_kernel_height) / conv1_7x7_s2_stride + 1);
const int conv1_7x7_s2_out_width_per_block = ((WIDTH_FEATURE_GLOBAL - conv1_7x7_s2_kernel_width) / conv1_7x7_s2_stride + 1);

const int conv1_7x7_s2_outer_height = (conv1_7x7_s2_out_height / conv1_7x7_s2_out_height_per_block) +
((conv1_7x7_s2_out_height % conv1_7x7_s2_out_height_per_block) == 0 ? 0 : 1);
const int conv1_7x7_s2_outer_width = (conv1_7x7_s2_out_width / conv1_7x7_s2_out_width_per_block) +
((conv1_7x7_s2_out_width % conv1_7x7_s2_out_width_per_block) == 0 ? 0 : 1);
const int conv1_7x7_s2_outer_out_channel = (conv1_7x7_s2_kernel_num / OUT_CHANNEL_WEIGHT_GLOBAL_7x7) + 
((conv1_7x7_s2_kernel_num % OUT_CHANNEL_WEIGHT_GLOBAL_7x7) == 0 ? 0 : 1);
const int conv1_7x7_s2_outer_in_channel = (conv1_7x7_s2_in_channel / CHANNEL_FEATURE_GLOBAL) + 
((conv1_7x7_s2_in_channel % CHANNEL_FEATURE_GLOBAL) == 0 ? 0 : 1);


const int conv1_7x7_s2_inner_height = (conv1_7x7_s2_out_height_per_block / OUT_HEIGHT_CONV7x7_S2) +
((conv1_7x7_s2_out_height_per_block % OUT_HEIGHT_CONV7x7_S2) == 0 ? 0 : 1);
const int conv1_7x7_s2_inner_width = (conv1_7x7_s2_out_width_per_block / OUT_WIDTH_CONV7x7_S2) +
((conv1_7x7_s2_out_width_per_block % OUT_WIDTH_CONV7x7_S2) == 0 ? 0 : 1);
const int conv1_7x7_s2_inner_out_channel = (OUT_CHANNEL_WEIGHT_GLOBAL_7x7 / NUM_PE_CONV7x7_S2) + ((OUT_CHANNEL_WEIGHT_GLOBAL_7x7 % NUM_PE_CONV7x7_S2) == 0 ? 0 : 1);
const int conv1_7x7_s2_inner_in_channel = ( CHANNEL_FEATURE_GLOBAL/ IN_CHAN_CONV7x7_S2) + ((CHANNEL_FEATURE_GLOBAL % IN_CHAN_CONV7x7_S2) == 0 ? 0 : 1);

const int conv1_7x7_s2_inner_pe_parallel = NUM_PE_CONV7x7_S2;

const int a[2] = { 1,32 };

#endif