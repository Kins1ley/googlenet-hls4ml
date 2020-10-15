#include <iostream>
#include <fstream>
#include <iomanip>
#include "pch.h"
#include "header.h"
#include "googlenet.h"

void software(
	FIX_INT20 inputs[pool1_3x3_s2_in_channel*pool1_3x3_s2_in_height*pool1_3x3_s2_in_width],
	FIX_INT20 outputs[pool1_3x3_s2_out_channel*pool1_3x3_s2_out_height*pool1_3x3_s2_out_width]
)
{
	for (int oh_idx = 0; oh_idx < pool1_3x3_s2_out_height; ++oh_idx) {
		for (int ow_idx = 0; ow_idx < pool1_3x3_s2_out_width; ++ow_idx) {
			for (int ic_idx = 0; ic_idx < pool1_3x3_s2_out_channel; ++ic_idx) {
				for (int kh_idx = 0; kh_idx < pool1_3x3_s2_kernel_height; ++kh_idx) {
					for (int kw_idx = 0; kw_idx < pool1_3x3_s2_kernel_width; ++kw_idx) {
						int AVGPOOL = 0;
						if (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top < 0) continue;
						if (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top >= pool1_3x3_s2_in_height) continue;
						if (ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left < 0) continue;
						if (ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left >= pool1_3x3_s2_in_width) continue;

						if ((kh_idx == 0 || (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top) == 0) && (kw_idx == 0 || (ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left) == 0)) {
							if (AVGPOOL) {
								outputs[ic_idx * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh_idx * pool1_3x3_s2_out_width + ow_idx] = (FIX_INT20)inputs[ic_idx * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width + (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top) * pool1_3x3_s2_in_width + ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left] / (FIX_INT20)(pool1_3x3_s2_kernel_height * pool1_3x3_s2_kernel_width);
							}
							else {
								outputs[ic_idx * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh_idx * pool1_3x3_s2_out_width + ow_idx] = (FIX_INT20)inputs[ic_idx * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width + (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top) * pool1_3x3_s2_in_width + ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left];

							}
						}
						else {
							if (AVGPOOL) {
								outputs[ic_idx * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh_idx * pool1_3x3_s2_out_width + ow_idx] += (FIX_INT20)inputs[ic_idx * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width + (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top) * pool1_3x3_s2_in_width + ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left] / (FIX_INT20)(pool1_3x3_s2_kernel_height * pool1_3x3_s2_kernel_width);
							}
							else {
								outputs[ic_idx * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh_idx * pool1_3x3_s2_out_width + ow_idx] = (FIX_INT20)inputs[ic_idx * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width + (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top) * pool1_3x3_s2_in_width + ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left] > outputs[ic_idx * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh_idx * pool1_3x3_s2_out_width + ow_idx] ? (FIX_INT20)inputs[ic_idx * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width + (oh_idx * pool1_3x3_s2_stride + kh_idx - pool1_3x3_s2_pad_top) * pool1_3x3_s2_in_width + ow_idx * pool1_3x3_s2_stride + kw_idx - pool1_3x3_s2_pad_left] : outputs[ic_idx * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh_idx * pool1_3x3_s2_out_width + ow_idx];
							}
						}
					}
				}
			}

		}
	}
}
int main(int argc, char const *argv[]) {

	FIX_INT20 inputs[pool1_3x3_s2_in_channel*pool1_3x3_s2_in_height*pool1_3x3_s2_in_width];
	FIX_INT20 outputs_software[pool1_3x3_s2_out_channel*pool1_3x3_s2_out_height*pool1_3x3_s2_out_width];
	FIX_INT20 outputs_hardware[pool1_3x3_s2_out_channel*pool1_3x3_s2_out_height*pool1_3x3_s2_out_width];
	srand(233);
	int show_results = 0;
	//inputs
	for (int ic = 0; ic < pool1_3x3_s2_in_channel; ++ic) {
		for (int ih = 0; ih < pool1_3x3_s2_in_height; ++ih) {
			for (int iw = 0; iw < pool1_3x3_s2_in_width; ++iw) {
				inputs[ic * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width+ih * pool1_3x3_s2_in_width +iw] = (FIX_INT20)(rand() % 6 );
				if(show_results)std::cout << inputs[ic * pool1_3x3_s2_in_height * pool1_3x3_s2_in_width + ih * pool1_3x3_s2_in_width + iw] << ",";
			}if(show_results)std::cout << std::endl;
		}if(show_results)std::cout << std::endl;
	}

	//initialize outputs
	for (int oc = 0; oc < pool1_3x3_s2_out_channel; ++oc) {
		for (int oh = 0; oh < pool1_3x3_s2_out_height; ++oh) {
			for (int ow = 0; ow < pool1_3x3_s2_out_width; ++ow) {
				outputs_hardware[oc * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width+oh * pool1_3x3_s2_out_width + ow] = 0;
			}
		}
	}

	googlenet(inputs, outputs_hardware);
	if(show_results)std::cout << "hardware outputs: " << std::endl;
	for (int oc = 0; oc < pool1_3x3_s2_out_channel; ++oc) {
		for (int oh = 0; oh < pool1_3x3_s2_out_height; ++oh) {
			for (int ow = 0; ow < pool1_3x3_s2_out_width; ++ow) {
				std::cout.width(4);
				if(show_results)std::cout << (float)outputs_hardware[oc * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh * pool1_3x3_s2_out_width + ow];
			}
			if(show_results)std::cout << std::endl;
		}
		if(show_results)std::cout << std::endl;
	}
	if(show_results)std::cout << std::endl;


	//initialize outputs
	for (int oc = 0; oc < pool1_3x3_s2_out_channel; ++oc) {
		for (int oh = 0; oh < pool1_3x3_s2_out_height; ++oh) {
			for (int ow = 0; ow < pool1_3x3_s2_out_width; ++ow) {
				outputs_software[oc * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh * pool1_3x3_s2_out_width + ow] = 0;
			}
		}
	}
	software(inputs,  outputs_software);
	if(show_results)std::cout << "software outputs: " << std::endl;
	for (int oc = 0; oc < pool1_3x3_s2_out_channel; ++oc) {
		for (int oh = 0; oh < pool1_3x3_s2_out_height; ++oh) {
			for (int ow = 0; ow < pool1_3x3_s2_out_width; ++ow) {
				std::cout.width(4);
				if(show_results)std::cout << (float)outputs_software[oc * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh * pool1_3x3_s2_out_width + ow];
			}
			if(show_results)std::cout << std::endl;
		}
		if(show_results)std::cout << std::endl;
	}
	//if(show_results)std::cout << std::endl;

	float diff = 0;
	for (int oc = 0; oc < pool1_3x3_s2_out_channel; ++oc) {
		for (int oh = 0; oh < pool1_3x3_s2_out_height; ++oh) {
			for (int ow = 0; ow < pool1_3x3_s2_out_width; ++ow) {
				diff += std::abs((float)outputs_software[oc * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh * pool1_3x3_s2_out_width + ow] - (float)outputs_hardware[oc * pool1_3x3_s2_out_height * pool1_3x3_s2_out_width + oh * pool1_3x3_s2_out_width + ow]);
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

