#include <iostream>
#include <fstream>
#include <iomanip>
#include "pch.h"
#include "header.h"
#include "googlenet.h"

void software(
	const FIX_INT20 inputs[conv1_7x7_s2_in_channel*conv1_7x7_s2_in_height*conv1_7x7_s2_in_width],
	const FIX_INT8 weights[conv1_7x7_s2_out_channel*conv1_7x7_s2_in_channel][conv1_7x7_s2_kernel_height][conv1_7x7_s2_kernel_width],
	const FIX_INT20 bias[conv1_7x7_s2_out_channel],
	FIX_INT20 outputs[conv1_7x7_s2_out_channel*conv1_7x7_s2_out_height*conv1_7x7_s2_out_width]
)
{
	for (int oh_idx = 0; oh_idx < conv1_7x7_s2_out_height; ++oh_idx) {
		for (int ow_idx = 0; ow_idx < conv1_7x7_s2_out_width; ++ow_idx) {
			for (int oc_idx = 0; oc_idx < conv1_7x7_s2_out_channel; ++oc_idx) {
				for (int ic_idx = 0; ic_idx < conv1_7x7_s2_in_channel; ++ic_idx) {
					if (ic_idx == 0) {
						outputs[oc_idx * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width +oh_idx * conv1_7x7_s2_out_width +ow_idx] = bias[oc_idx];
					}
					for (int kh_idx = 0; kh_idx < conv1_7x7_s2_kernel_height; ++kh_idx) {
						for (int kw_idx = 0; kw_idx < conv1_7x7_s2_kernel_width; ++kw_idx) {
							if (oh_idx*STRIDE_CONV1x1_S1 + kh_idx - conv1_7x7_s2_pad_top < 0) continue;
							if (oh_idx*STRIDE_CONV1x1_S1 + kh_idx - conv1_7x7_s2_pad_top >= conv1_7x7_s2_in_height) continue;
							if (ow_idx*STRIDE_CONV1x1_S1 + kw_idx - conv1_7x7_s2_pad_left < 0) continue;
							if (ow_idx*STRIDE_CONV1x1_S1 + kw_idx - conv1_7x7_s2_pad_left >= conv1_7x7_s2_in_width) continue;
							outputs[oc_idx * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh_idx * conv1_7x7_s2_out_width + ow_idx] += (FIX_INT20)(inputs[ic_idx * conv1_7x7_s2_in_height * conv1_7x7_s2_in_width+(oh_idx*STRIDE_CONV1x1_S1 + kh_idx - conv1_7x7_s2_pad_top) * conv1_7x7_s2_in_width +ow_idx*STRIDE_CONV1x1_S1 + kw_idx - conv1_7x7_s2_pad_left]
								* weights[oc_idx *conv1_7x7_s2_in_channel +ic_idx][kh_idx][kw_idx]);
						}
					}
				}
				outputs[oc_idx * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh_idx * conv1_7x7_s2_out_width + ow_idx] = outputs[oc_idx * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh_idx * conv1_7x7_s2_out_width + ow_idx] > 0 ? outputs[oc_idx * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh_idx * conv1_7x7_s2_out_width + ow_idx] : 0;
			}
		}
	}
}

int main(int argc, char const *argv[]) {

	FIX_INT20 inputs[conv1_7x7_s2_in_channel*conv1_7x7_s2_in_height*conv1_7x7_s2_in_width];
	FIX_INT8 weights[conv1_7x7_s2_out_channel*conv1_7x7_s2_in_channel][conv1_7x7_s2_kernel_height][conv1_7x7_s2_kernel_width];
	FIX_INT20 outputs_software[conv1_7x7_s2_out_channel*conv1_7x7_s2_out_height*conv1_7x7_s2_out_width];
	FIX_INT20 outputs_hardware[conv1_7x7_s2_out_channel*conv1_7x7_s2_out_height*conv1_7x7_s2_out_width];
	FIX_INT20 bias[conv1_7x7_s2_bias_num];
	int show_results = 0;
	srand(233);
	//inputs
	for (int ic = 0; ic < conv1_7x7_s2_in_channel; ++ic) {
		for (int ih = 0; ih < conv1_7x7_s2_in_height; ++ih) {
			for (int iw = 0; iw < conv1_7x7_s2_in_width; ++iw) {
				inputs[ic * conv1_7x7_s2_in_height * conv1_7x7_s2_in_width+ih * conv1_7x7_s2_in_width +iw] = (FIX_INT20)(rand() % 6 - 3);
				if(show_results)std::cout << inputs[ic * conv1_7x7_s2_in_height * conv1_7x7_s2_in_width + ih * conv1_7x7_s2_in_width + iw] << ",";
			}if (show_results)std::cout << std::endl;
		}if (show_results)std::cout << std::endl;
	}
	//weights
	for (int o_idx = 0; o_idx < conv1_7x7_s2_out_channel; ++o_idx) {
		bias[o_idx] = (FIX_INT8)(rand() % 6 - 3);
		if (show_results)std::cout << "bias " << bias[o_idx] << std::endl;
		for (int i_idx = 0; i_idx < conv1_7x7_s2_in_channel; ++i_idx) {
			for (int h_idx = 0; h_idx < conv1_7x7_s2_kernel_height; ++h_idx) {
				for (int w_idx = 0; w_idx < conv1_7x7_s2_kernel_width; ++w_idx) {
					weights[o_idx * conv1_7x7_s2_in_channel + i_idx][h_idx][w_idx] = (FIX_INT8)(rand() % 6 - 3);
					if (show_results)std::cout << weights[o_idx * conv1_7x7_s2_in_channel + i_idx][h_idx][w_idx] << ",";
				}if (show_results)std::cout << std::endl;
			}if (show_results)std::cout << std::endl;
		}if (show_results)std::cout << std::endl;
	}
	//initialize outputs
	for (int oc = 0; oc < conv1_7x7_s2_out_channel; ++oc) {
		for (int oh = 0; oh < conv1_7x7_s2_out_height; ++oh) {
			for (int ow = 0; ow < conv1_7x7_s2_out_width; ++ow) {
				outputs_hardware[oc* conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh * conv1_7x7_s2_out_width + ow] = 0;
			}
		}
	}
	googlenet(inputs, weights, bias, outputs_hardware);
	if (show_results)std::cout << "hardware outputs: " << std::endl;
	for (int oc = 0; oc < conv1_7x7_s2_out_channel; ++oc) {
		for (int oh = 0; oh < conv1_7x7_s2_out_height; ++oh) {
			for (int ow = 0; ow < conv1_7x7_s2_out_width; ++ow) {
				std::cout.width(4);
				if (show_results)std::cout << (float)outputs_hardware[oc * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh * conv1_7x7_s2_out_width + ow];
			}
			if (show_results)std::cout << std::endl;
		}
		if (show_results)std::cout << std::endl;
	}
	if (show_results)std::cout << std::endl;


	//initialize outputs
	for (int oc = 0; oc < conv1_7x7_s2_out_channel; ++oc) {
		for (int oh = 0; oh < conv1_7x7_s2_out_height; ++oh) {
			for (int ow = 0; ow < conv1_7x7_s2_out_width; ++ow) {
				outputs_software[oc * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh * conv1_7x7_s2_out_width + ow] = 0;
			}
		}
	}
	software(inputs, weights, bias, outputs_software);
	if (show_results)std::cout << "software outputs: " << std::endl;
	for (int oc = 0; oc < conv1_7x7_s2_out_channel; ++oc) {
		for (int oh = 0; oh < conv1_7x7_s2_out_height; ++oh) {
			for (int ow = 0; ow < conv1_7x7_s2_out_width; ++ow) {
				std::cout.width(4);
				if (show_results)std::cout << (float)outputs_software[oc * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh * conv1_7x7_s2_out_width + ow];
			}
			if (show_results)std::cout << std::endl;
		}
		if (show_results)std::cout << std::endl;
	}
	//std::cout << std::endl;

	float diff = 0;
	for (int oc = 0; oc < conv1_7x7_s2_out_channel; ++oc) {
		for (int oh = 0; oh < conv1_7x7_s2_out_height; ++oh) {
			for (int ow = 0; ow < conv1_7x7_s2_out_width; ++ow) {
				diff += std::abs((float)outputs_software[oc * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh * conv1_7x7_s2_out_width + ow] - (float)outputs_hardware[oc * conv1_7x7_s2_out_height * conv1_7x7_s2_out_width + oh * conv1_7x7_s2_out_width + ow]);
			}
		}
	}
	if (diff < 0.1) {
		if (show_results)std::cout << "software outputs match hardware outputs! " << std::endl;
		std::ofstream outfile;
		outfile.open("validate_success.txt", std::ios::out); 
		outfile << "validate_success" << std::endl;
		outfile.close();
	}
	else {
		if (show_results)std::cout << "software outputs do not match hardware outputs! " << std::endl;
		std::ofstream outfile;
		outfile.open("validate_fail.txt", std::ios::out);
		outfile << "validate_fail" << std::endl;
		outfile.close();
	}

	return 0;
}

