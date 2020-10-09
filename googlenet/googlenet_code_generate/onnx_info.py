from __future__ import print_function
import numpy as np
import os
import math
from onnx import ModelProto, GraphProto, NodeProto, TensorProto
from onnx import optimizer, helper, numpy_helper, shape_inference
import argparse

parser = argparse.ArgumentParser(description='onnx reader')
parser.add_argument("--fname",type=str,default="googlenet-7.onnx")
args=parser.parse_args()
MAXMULT = 4096
def get_onnx_attribute(operation, name, default=None):
    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value
def get_input_shape(model, operation, input_idx=0):
    for x in model.graph.input :
        if operation.input[input_idx]==x.name:
            return [d.dim_value for d in x.type.tensor_type.shape.dim]
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.input[input_idx]), 0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]
def get_output_shape(model, operation, output_idx=0):
    for x in model.graph.output :
        if operation.output[output_idx]==x.name:
            return [d.dim_value for d in x.type.tensor_type.shape.dim]
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.output[output_idx]),0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]

def sanitize_input_name(name):
    return name.replace("/","_")

def read_config(file_name):

    # Extract model architecture
    model = ModelProto()
    with open(file_name, 'rb') as fid:
        model.ParseFromString(fid.read())
    passes = ['fuse_transpose_into_gemm', 'fuse_matmul_add_bias_into_gemm', 'eliminate_nop_transpose',
              'fuse_consecutive_transposes']
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    model = optimizer.optimize(model, passes)
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model

    all_layer_config = []
    layer_config_dict_base = {"left_bracket": "{", "right_bracket": "}"}
    for operation in model.graph.node:
        #print(operation.op_type)
        if operation.op_type== "Conv":
            layer_config_dict = layer_config_dict_base.copy()
            layer_config_dict["conv_layer_name"]="{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["conv_layer_type"]="CONV{kernel_shape}x{kernel_shape}_S{stride}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["global_weight_name"]="global_weight_{kernel_shape}x{kernel_shape}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["global_weight_type"]="WEIGHT_GLOBAL_{kernel_shape}x{kernel_shape}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["conv_pe_name"]="conv_output_reuse{kernel_shape}x{kernel_shape}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["conv_layer_input"]="{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["conv_layer_output"]="{output}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["DDR_weight_type"]="DDR_weight_{kernel_shape}x{kernel_shape}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            all_layer_config.append(layer_config_dict)
        if operation.op_type== "MaxPool":
            layer_config_dict = layer_config_dict_base.copy()
            layer_config_dict["pool_layer_name"] = "{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_layer_type"] = "MAXPOOL{kernel_shape}x{kernel_shape}_S{stride}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_pe_name"] = "pool{kernel_shape}x{kernel_shape}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_layer_input"] = "{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_layer_output"] = "{output}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            all_layer_config.append(layer_config_dict)
        if operation.op_type== "AveragePool":
            layer_config_dict = layer_config_dict_base.copy()
            layer_config_dict["pool_layer_name"] = "{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_layer_type"] = "AVGPOOL{kernel_shape}x{kernel_shape}_S{stride}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_pe_name"] = "pool{kernel_shape}x{kernel_shape}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_layer_input"] = "{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pool_layer_output"] = "{output}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  kernel_shape=get_onnx_attribute(operation,"kernel_shape")[0],
                  stride=get_onnx_attribute(operation, "strides")[0],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            all_layer_config.append(layer_config_dict)
        if operation.op_type== "LRN":
            layer_config_dict = layer_config_dict_base.copy()
            layer_config_dict["lrn_layer_name"] = "{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["lrn_layer_type"] = "LRN"
            layer_config_dict["lrn_pe_name"] = "LRN"
            layer_config_dict["lrn_layer_input"] = "{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            layer_config_dict["lrn_layer_output"] = "{output}".format(name=sanitize_input_name(operation.output[0])[:-2],
                  input=sanitize_input_name(operation.input[0]),
                  output=sanitize_input_name(operation.output[0]))
            all_layer_config.append(layer_config_dict)
        if operation.op_type == "Concat":
            #replace the output name of concatenated layers
            inputs=[sanitize_input_name(i) for i in operation.input]
            outputs=[sanitize_input_name(i) for i in operation.output]
            for i in range(len(all_layer_config)):
                for key in all_layer_config[i].keys():
                    if "layer_output" in key:
                        #if all_layer_config[i][key] in inputs:
                        for input_name in inputs:
                            if all_layer_config[i][key][:-2] in input_name:
                                #print("replacing {} with {}".format(all_layer_config[i][key],outputs[0]))
                                all_layer_config[i][key]=outputs[0]

    return all_layer_config

header_template_conv="///layer {layer_name}\n\
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
const int {layer_name}_out_channel_DDR_offset = {out_channel_DDR_offset};\n"

header_template_pool="///layer {layer_name}\n\
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
const int {layer_name}_out_channel_DDR_offset = {out_channel_DDR_offset};\n"

header_template_lrn="///layer {layer_name}\n\
const int {layer_name}_in_channel = {in_channel};\n\
const int {layer_name}_in_height = {in_height};\n\
const int {layer_name}_in_width = {in_width};\n\
const int {layer_name}_out_channel = {layer_name}_in_channel;\n\
const int {layer_name}_out_height = {layer_name}_in_height;\n\
const int {layer_name}_out_width = {layer_name}_in_width;\n\
const int {layer_name}_depth_radius = {depth_radius};\n"

def read_onnx(file_name):

    # Extract model architecture
    model = ModelProto()
    with open(file_name, 'rb') as fid:
        model.ParseFromString(fid.read())
    passes = ['fuse_transpose_into_gemm', 'fuse_matmul_add_bias_into_gemm', 'eliminate_nop_transpose',
              'fuse_consecutive_transposes']
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    model = optimizer.optimize(model, passes)
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model

    header_dict = {}

    for operation in model.graph.node:
        #print(operation.op_type)
        if operation.op_type== "Conv":
            input_shape = get_input_shape(model, operation)
            output_shape = get_output_shape(model, operation)
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')
            pads = get_onnx_attribute(operation, 'pads')
            layer_name = sanitize_input_name(operation.output[0])[:-2]
            in_channel=input_shape[1]
            in_height=input_shape[2]
            in_width=input_shape[3]
            pad_top=pads[0]
            pad_bottom=pads[1]
            pad_left=pads[2]
            pad_right=pads[3]
            kernel_num=output_shape[1]
            kernel_height=kernel_shape[0]
            kernel_width=kernel_shape[1]
            out_height=output_shape[2]
            out_width=output_shape[3]
            stride=strides[0]
            kernel_channel_DDR_offset=0
            bias_DDR_offset=0
            out_channel_DDR_offset=0
            header_dict[layer_name]=header_template_conv.format(layer_name=layer_name,in_channel=in_channel,
                                                                in_height=in_height,in_width=in_width,
                                                                pad_top=pad_top,pad_bottom=pad_bottom,
                                                                pad_left=pad_left,pad_right=pad_right,
                                                                kernel_num=kernel_num,kernel_height=kernel_height,
                                                                kernel_width=kernel_width,out_height=out_height,
                                                                out_width=out_width,stride=stride,
                                                                kernel_channel_DDR_offset=kernel_channel_DDR_offset,
                                                                bias_DDR_offset=bias_DDR_offset,
                                                                out_channel_DDR_offset=out_channel_DDR_offset)
        if operation.op_type== "MaxPool":
            input_shape = get_input_shape(model, operation)
            output_shape = get_output_shape(model, operation)
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')
            pads = get_onnx_attribute(operation, 'pads')
            layer_name = sanitize_input_name(operation.output[0])[:-2]
            in_channel = input_shape[1]
            in_height = input_shape[2]
            in_width = input_shape[3]
            pad_top = pads[0]
            pad_bottom = pads[1]
            pad_left = pads[2]
            pad_right = pads[3]
            kernel_num = output_shape[1]
            kernel_height = kernel_shape[0]
            kernel_width = kernel_shape[1]
            out_height = output_shape[2]
            out_width = output_shape[3]
            stride = strides[0]
            out_channel_DDR_offset = 0
            header_dict[layer_name]=header_template_pool.format(layer_name=layer_name, in_channel=in_channel,
                                                                in_height=in_height, in_width=in_width,
                                                                pad_top=pad_top, pad_bottom=pad_bottom,
                                                                pad_left=pad_left, pad_right=pad_right,
                                                                kernel_num=kernel_num, kernel_height=kernel_height,
                                                                kernel_width=kernel_width, out_height=out_height,
                                                                out_width=out_width, stride=stride,
                                                                out_channel_DDR_offset=out_channel_DDR_offset)

        if operation.op_type== "AveragePool":
            input_shape = get_input_shape(model, operation)
            output_shape = get_output_shape(model, operation)
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')
            pads = get_onnx_attribute(operation, 'pads')
            layer_name = sanitize_input_name(operation.output[0])[:-2]
            in_channel = input_shape[1]
            in_height = input_shape[2]
            in_width = input_shape[3]
            pad_top = pads[0]
            pad_bottom = pads[1]
            pad_left = pads[2]
            pad_right = pads[3]
            kernel_num = output_shape[1]
            kernel_height = kernel_shape[0]
            kernel_width = kernel_shape[1]
            out_height = output_shape[2]
            out_width = output_shape[3]
            stride = strides[0]
            out_channel_DDR_offset = 0
            header_dict[layer_name]=header_template_pool.format(layer_name=layer_name, in_channel=in_channel,
                                                                in_height=in_height, in_width=in_width,
                                                                pad_top=pad_top, pad_bottom=pad_bottom,
                                                                pad_left=pad_left, pad_right=pad_right,
                                                                kernel_num=kernel_num, kernel_height=kernel_height,
                                                                kernel_width=kernel_width, out_height=out_height,
                                                                out_width=out_width, stride=stride,
                                                                out_channel_DDR_offset=out_channel_DDR_offset)

        if operation.op_type== "LRN":
            input_shape = get_input_shape(model, operation)
            layer_name = sanitize_input_name(operation.output[0])[:-2]
            depth_radius=get_onnx_attribute(operation,"size")
            in_channel = input_shape[1]
            in_height = input_shape[2]
            in_width = input_shape[3]

            header_dict[layer_name]=header_template_lrn.format(layer_name=layer_name, in_channel=in_channel,
                                                                in_height=in_height, in_width=in_width,
                                                                depth_radius=depth_radius)

        if operation.op_type == "Concat":
            continue


    return header_dict

if __name__=="__main__":
    f_name=args.fname
    result=read_onnx(file_name=f_name)
    print(len(result))
    for i in result:
        print(i)
