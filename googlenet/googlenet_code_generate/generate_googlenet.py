import os
import shutil
import requests
import onnx_info
import generate_lrn
import generate_pool
import generate_conv

max_layer_to_write = 1000
onnx_file_name = "googlenet-7.onnx"
if not os.path.exists(onnx_file_name):
    url = 'https://github.com/onnx/models/blob/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.onnx'
    raise FileNotFoundError("onnx file not found, please download it at\n{}".format(url))
    print("downloading googlenet-7.onnx")
    r = requests.get(url, stream=True)
    with open(onnx_file_name, "wb") as Pypdf:
        for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
            if chunk:
                Pypdf.write(chunk)
layer_configs = onnx_info.read_onnx(onnx_file_name)
onnx_info.allocate_DRAM_feature(layer_configs)
onnx_info.read_value(onnx_file_name,layer_configs,write_to_files=True)
out_path = "./out_code/"
template_path = "./template_code/"
files = os.listdir(template_path)
onnx_info.setDir(out_path)
shutil.move("weights",os.path.join(out_path,"weights"))
print("files to process:", files)


# print(layer_configs)
def write_testbench(template_file_name, out_file_name, configs):
    insert_tag_DRAM = "/////DRAM_insert/////"
    insert_tag_param = "/////param_insert/////"
    template_file = open(template_file_name, "r")
    out_file = open(out_file_name, "w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text, file=out_file, end="")
        if insert_tag_DRAM in text:
            continue  # no need to add DRAM ports for features
            """
            DRAM_param_dict= {}
            for config in configs:
                if "conv_layer_name" in config.keys():
                    layer_name=config["conv_layer_name"]
                    layer_output=config["layer_output_name"]
                if "pool_layer_name" in config.keys():
                    layer_name = config["pool_layer_name"]
                    layer_output=config["layer_output_name"]
                if "lrn_layer_name" in config.keys():
                    layer_name = config["lrn_layer_name"]
                    layer_output=config["layer_output_name"]
                if layer_output not in DRAM_param_dict.keys():
                    DRAM_param_dict[layer_output]="static FIX_INT20 {layer_output}[{layer_name}_out_channel][{layer_name}_out_height][{layer_name}_out_width];".format(
                        layer_output=layer_output,layer_name=layer_name)
                else:
                    DRAM_param_dict[layer_output]=DRAM_param_dict[layer_output].replace("_out_channel]","_out_channel+{layer_name}_out_channel]").format(
                        layer_output=layer_output,layer_name=layer_name)
            for key in DRAM_param_dict.keys():
                print(DRAM_param_dict[key],file=out_file)"""
        elif insert_tag_param in text:
            continue  # no need to add DRAM ports for features
            """
            DRAM_interface_dict = {}
            for config in configs:
                if "conv_layer_name" in config.keys():
                    layer_name = config["conv_layer_name"]
                    layer_output = config["layer_output_name"]
                if "pool_layer_name" in config.keys():
                    layer_name = config["pool_layer_name"]
                    layer_output = config["layer_output_name"]
                if "lrn_layer_name" in config.keys():
                    layer_name = config["lrn_layer_name"]
                    layer_output = config["layer_output_name"]
                if layer_output not in DRAM_interface_dict.keys():
                    DRAM_interface_dict[layer_output]="{layer_output}".format(
                        layer_output=layer_output, layer_name=layer_name)
            for key in DRAM_interface_dict.keys():
                print(DRAM_interface_dict[key],"," if DRAM_interface_dict[key]!= "out" else "",file=out_file)
            """


def write_googlenet(template_file_name, out_file_name, configs):
    top_function_code_conv = generate_conv.generate_conv(part="top_function")
    top_function_code_pool = generate_pool.generate_pool(part="top_function")
    top_function_code_lrn = generate_lrn.generate_lrn(part="top_function")
    insert_tag_top_function = "/////top_function_insert/////"
    template_file = open(template_file_name, "r")
    out_file = open(out_file_name, "w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text, file=out_file, end="")

        if insert_tag_top_function in text:
            layer_num = 0
            for config in configs:
                if layer_num >= max_layer_to_write:
                    break
                if "conv" in config["layer_type"].lower():
                    print(top_function_code_conv.format(**config), file=out_file)
                if "pool" in config["layer_type"].lower():
                    print(top_function_code_pool.format(**config), file=out_file)
                if "lrn" in config["layer_type"].lower():
                    print(top_function_code_lrn.format(**config), file=out_file)
                layer_num += 1


def write_template_config(template_file_name, out_file_name, configs):
    insert_tag = "/////template_config_insert/////"
    template_file = open(template_file_name, "r")
    out_file = open(out_file_name, "w")
    layer_output_list = []
    template_code_conv = generate_conv.generate_conv(part="template_config")
    template_code_pool = generate_pool.generate_pool(part="template_config")
    template_code_lrn = generate_lrn.generate_lrn(part="template_config")
    DDR_weight_config_template = "struct DDR_weight_{layer_name}_config : nnet::Weight_Memory {{\n\
	typedef FIX_INT20 weight_type;\n\
	static const unsigned in_channel = {layer_name}_kernel_channel;\n\
	static const unsigned out_channel = {layer_name}_kernel_num;\n\
	static const unsigned height = {layer_name}_kernel_height;\n\
	static const unsigned width = {layer_name}_kernel_width;\n\
}};"
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text, file=out_file, end="")
        if insert_tag in text:
            for config in configs:
                # print(config[layer_output_key])
                if "conv" in config["layer_type"].lower():
                    print(DDR_weight_config_template.format(**config), file=out_file)
                if config["layer_output_name"] in layer_output_list:
                    # print("ignoring existing output {}".format(config[layer_output_key]))
                    continue
                else:
                    layer_output_list.append(config["layer_output_name"])
                if "conv" in config["layer_type"].lower():
                    print(template_code_conv.format(**config), file=out_file)
                if "pool" in config["layer_type"].lower():
                    print(template_code_pool.format(**config), file=out_file)
                if "lrn" in config["layer_type"].lower():
                    print(template_code_lrn.format(**config), file=out_file)


def write_allocate_config(template_file_name, out_file_name, configs):
    insert_tag = "/////allocate_config_insert/////"
    template_file = open(template_file_name, "r")
    out_file = open(out_file_name, "w")
    allocate_config_code_conv = generate_conv.generate_conv(part="allocate_config")
    allocate_config_code_pool = generate_pool.generate_pool(part="allocate_config")
    allocate_config_code_lrn = generate_lrn.generate_lrn(part="allocate_config")

    while True:
        text = template_file.readline()
        if not text:
            break
        print(text, file=out_file, end="")
        if insert_tag in text:
            for config in configs:
                if "conv" in config["layer_type"].lower():
                    print(allocate_config_code_conv.format(**config), file=out_file)
                if "pool" in config["layer_type"].lower():
                    print(allocate_config_code_pool.format(**config), file=out_file)
                if "lrn" in config["layer_type"].lower():
                    print(allocate_config_code_lrn.format(**config), file=out_file)


def write_header(template_file_name, out_file_name, configs):
    header_template_conv = "///layer {layer_name}\n\
    const int {layer_name}_in_channel = {in_channel};\n\
    const int {layer_name}_in_height = {in_height};\n\
    const int {layer_name}_in_width = {in_width};\n\
    const int {layer_name}_pad_top = {pad_top};\n\
    const int {layer_name}_pad_bottom = {pad_bottom};\n\
    const int {layer_name}_pad_left = {pad_left};\n\
    const int {layer_name}_pad_right = {pad_right};\n\
    const int {layer_name}_kernel_num = {kernel_num};\n\
    const int {layer_name}_kernel_channel = {layer_name}_in_channel;\n\
    const int {layer_name}_kernel_channel_DDR_offset = {kernel_channel_DDR_offset};\n\
    const int {layer_name}_kernel_height = {kernel_height};\n\
    const int {layer_name}_kernel_width = {kernel_width};\n\
    const int {layer_name}_bias_num = {layer_name}_kernel_num;\n\
    const int {layer_name}_bias_DDR_offset = {bias_DDR_offset};\n\
    const int {layer_name}_out_channel = {layer_name}_kernel_num;\n\
    const int {layer_name}_out_height = {out_height};\n\
    const int {layer_name}_out_width = {out_width};\n\
    const int {layer_name}_stride = {stride};\n\
    const int {layer_name}_out_feature_DDR_offset = {out_feature_DDR_offset};\n"

    header_template_pool = "///layer {layer_name}\n\
    const int {layer_name}_in_channel = {in_channel};\n\
    const int {layer_name}_in_height = {in_height};\n\
    const int {layer_name}_in_width = {in_width};\n\
    const int {layer_name}_pad_top = {pad_top};\n\
    const int {layer_name}_pad_bottom = {pad_bottom};\n\
    const int {layer_name}_pad_left = {pad_left};\n\
    const int {layer_name}_pad_right = {pad_right};\n\
    const int {layer_name}_out_channel = {layer_name}_in_channel;\n\
    const int {layer_name}_kernel_height = {kernel_height};\n\
    const int {layer_name}_kernel_width = {kernel_width};\n\
    const int {layer_name}_out_height = {out_height};\n\
    const int {layer_name}_out_width = {out_width};\n\
    const int {layer_name}_stride = {stride};\n\
    const int {layer_name}_out_feature_DDR_offset = {out_feature_DDR_offset};\n"

    header_template_lrn = "///layer {layer_name}\n\
    const int {layer_name}_in_channel = {in_channel};\n\
    const int {layer_name}_in_height = {in_height};\n\
    const int {layer_name}_in_width = {in_width};\n\
    const int {layer_name}_out_channel = {layer_name}_in_channel;\n\
    const int {layer_name}_out_height = {layer_name}_in_height;\n\
    const int {layer_name}_out_width = {layer_name}_in_width;\n\
    const int {layer_name}_depth_radius = {depth_radius};\n"
    insert_tag = "/////header_insert/////"
    template_file = open(template_file_name, "r")
    out_file = open(out_file_name, "w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text, file=out_file, end="")
        if insert_tag in text:
            for config in configs:
                if "conv" in config["layer_type"].lower():
                    print(header_template_conv.format(**config), file=out_file)
                if "pool" in config["layer_type"].lower():
                    print(header_template_pool.format(**config), file=out_file)
                if "lrn" in config["layer_type"].lower():
                    print(header_template_lrn.format(**config), file=out_file)


def write_other(template_file_name, out_file_name):
    template_file = open(template_file_name, "r")
    out_file = open(out_file_name, "w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text, file=out_file, end="")


for fn in files:
    template_file_name = os.path.join(template_path, fn)
    out_file_name = os.path.join(out_path, fn)
    if "googlenet.cpp" in fn:
        write_testbench(template_file_name, out_file_name, layer_configs)
    elif "googlenet.h" in fn:
        write_googlenet(template_file_name, out_file_name, layer_configs)
    elif "template_config.h" in fn:
        write_template_config(template_file_name, out_file_name, layer_configs)
    elif "allocate_config.h" in fn:
        write_allocate_config(template_file_name, out_file_name, layer_configs)
    elif "header.h" in fn:
        write_header(template_file_name, out_file_name, layer_configs)
    else:
        write_other(template_file_name, out_file_name)
