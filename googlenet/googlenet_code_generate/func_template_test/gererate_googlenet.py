import os
import requests
import onnx_info
import generate_lrn
import generate_pool
import generate_conv
max_layer_to_write=1000
onnx_file_name="googlenet-7.onnx"
if not os.path.exists(onnx_file_name) :
    url = 'https://github.com/onnx/models/blob/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.onnx'
    raise FileExistsError("onnx file not found, please download it at\n{}".format(url))
    print("downloading googlenet-7.onnx")
    r = requests.get(url, stream=True)
    with open(onnx_file_name, "wb") as Pypdf:
        for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
            if chunk:
                Pypdf.write(chunk)
layer_configs=onnx_info.read_config(onnx_file_name)
headers=onnx_info.read_onnx(onnx_file_name)

out_path = "./out_code/"
template_path="./template_code/"
files= os.listdir(template_path)
print("files to process:",files)
#print(layer_configs)
def write_testbench(template_file_name,out_file_name,configs,headers):
    insert_tag_DRAM = "/////DRAM_insert/////"
    insert_tag_param = "/////param_insert/////"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag_DRAM in text:
            continue# no need to add DRAM ports for features
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
                print(DRAM_param_dict[key],file=out_file)
        elif insert_tag_param in text:
            continue# no need to add DRAM ports for features
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


def write_googlenet(template_file_name,out_file_name,configs,headers):
    insert_tag_DRAM = "/////DRAM_insert/////"
    insert_tag_interface = "/////interface_insert/////"
    insert_tag_top_function="/////top_function_insert/////"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag_DRAM in text:
            continue# no need to add DRAM ports for features
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
                    DRAM_param_dict[layer_output]="FIX_INT20 {layer_output}[{layer_name}_out_channel][{layer_name}_out_height][{layer_name}_out_width],".format(
                        layer_output=layer_output,layer_name=layer_name)
                else:
                    DRAM_param_dict[layer_output]=DRAM_param_dict[layer_output].replace("_out_channel]","_out_channel+{layer_name}_out_channel]").format(
                        layer_output=layer_output,layer_name=layer_name)
            for key in DRAM_param_dict.keys():
                if key !="out":
                    print(DRAM_param_dict[key],file=out_file)
                else:
                    print(DRAM_param_dict[key][:-1], file=out_file)
        elif insert_tag_interface in text:
            continue# no need to add DRAM ports for features
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
                    DRAM_interface_dict[layer_output]="#pragma HLS INTERFACE m_axi depth=({layer_name}_out_channel)*{layer_name}_out_height*{layer_name}_out_width				port={layer_output}			offset=slave bundle=INPUT".format(
                        layer_output=layer_output, layer_name=layer_name)
                else:
                    DRAM_interface_dict[layer_output]=DRAM_interface_dict[layer_output].replace("_out_channel)","_out_channel+{layer_name}_out_channel)").format(
                        layer_output=layer_output,layer_name=layer_name)
            for key in DRAM_interface_dict.keys():
                print(DRAM_interface_dict[key],file=out_file)
        elif insert_tag_top_function in text:
            layer_num=0
            for config in configs:
                if layer_num>=max_layer_to_write:
                    break
                if "conv_layer_name" in config.keys():
                    generate_conv.generate_conv(config,out_file=out_file,part="top_function")
                if "pool_layer_name" in config.keys():
                    generate_pool.generate_pool(config,out_file=out_file,part="top_function")
                if "lrn_layer_name" in config.keys():
                    generate_lrn.generate_lrn(config,out_file=out_file,part="top_function")
                layer_num += 1

def write_template_config(template_file_name,out_file_name,configs):
    insert_tag="/////template_config_insert/////"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    layer_output_list=[]
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag in text:
            for config in configs:
                for key in config.keys():
                    if "layer_output" in key:
                        layer_output_key=key
                #print(config[layer_output_key])
                if config[layer_output_key] in layer_output_list:
                    #print("ignoring existing output {}".format(config[layer_output_key]))
                    continue
                else:
                    layer_output_list.append(config[layer_output_key])
                if "conv_layer_name" in config.keys():
                    generate_conv.generate_conv(config,out_file=out_file,part="template_config")
                if "pool_layer_name" in config.keys():
                    generate_pool.generate_pool(config,out_file=out_file,part="template_config")
                if "lrn_layer_name" in config.keys():
                    generate_lrn.generate_lrn(config,out_file=out_file,part="template_config")

def write_allocate_config(template_file_name,out_file_name,configs):
    insert_tag="/////allocate_config_insert/////"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag in text:
            for config in configs:
                if "conv_layer_name" in config.keys():
                    generate_conv.generate_conv(config,out_file=out_file,part="allocate_config")
                if "pool_layer_name" in config.keys():
                    generate_pool.generate_pool(config,out_file=out_file,part="allocate_config")
                if "lrn_layer_name" in config.keys():
                    generate_lrn.generate_lrn(config,out_file=out_file,part="allocate_config")

def write_header(template_file_name,out_file_name,headers):
    insert_tag="/////header_insert/////"
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
        if insert_tag in text:
            for key in headers.keys():
                print(headers[key],file=out_file)

def write_other(template_file_name,out_file_name):
    template_file=open(template_file_name,"r")
    out_file=open(out_file_name,"w")
    while True:
        text = template_file.readline()
        if not text:
            break
        print(text,file=out_file,end="")
for fn in files:
    template_file_name=os.path.join(template_path,fn)
    out_file_name=os.path.join(out_path,fn)
    if "googlenet.cpp" in fn:
        write_testbench(template_file_name,out_file_name,layer_configs,headers)
    elif "googlenet.h" in fn:
        write_googlenet(template_file_name,out_file_name,layer_configs,headers)
    elif "template_config.h" in fn:
        write_template_config(template_file_name, out_file_name, layer_configs)
    elif "allocate_config.h" in fn:
        write_allocate_config(template_file_name, out_file_name, layer_configs)
    elif "header.h" in fn:
        write_header(template_file_name, out_file_name, headers)
    else:
        write_other(template_file_name,out_file_name)
