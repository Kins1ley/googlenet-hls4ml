import os
import onnx_info
import generate_lrn
import generate_pool
import generate_conv

onnx_file_name="googlenet-7.onnx"
layer_configs=onnx_info.read_onnx(onnx_file_name)
print(len(layer_configs))

out_path = "./out_code/"
template_path="./template_code/"
files= os.listdir(template_path)
print(files)
#print(layer_configs)
def write_googlenet(template_file_name,out_file_name,configs):
    insert_tag="/////top_function_insert/////"
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
                    generate_conv.generate_conv(config,out_file=out_file,part="top_function")
                if "pool_layer_name" in config.keys():
                    generate_pool.generate_pool(config,out_file=out_file,part="top_function")
                if "lrn_layer_name" in config.keys():
                    generate_lrn.generate_lrn(config,out_file=out_file,part="top_function")

def write_template_config(template_file_name,out_file_name,configs):
    insert_tag="/////template_config_insert/////"
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
    if "googlenet.h" in fn:
        write_googlenet(template_file_name,out_file_name,layer_configs)
    elif "template_config.h" in fn:
        write_template_config(template_file_name, out_file_name, layer_configs)
    elif "allocate_config.h" in fn:
        write_allocate_config(template_file_name, out_file_name, layer_configs)
    else:
        write_other(template_file_name,out_file_name)
