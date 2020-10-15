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

def get_layer_list(file_name,layer_type):
    # Extract model architecture
    model = ModelProto()
    with open(file_name, 'rb') as fid:
        model.ParseFromString(fid.read())
    passes = ['fuse_transpose_into_gemm', 'fuse_matmul_add_bias_into_gemm', 'eliminate_nop_transpose',
              'fuse_consecutive_transposes']
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    model = optimizer.optimize(model, passes)
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    layer_list=[]
    for operation in model.graph.node:
        #print(operation.op_type)
        layer_config_dict={}
        if operation.op_type== "Conv" and layer_type.lower()=="conv":
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
            pad_bottom=pads[2]
            pad_left=pads[1]
            pad_right=pads[3]
            kernel_num=output_shape[1]
            kernel_height=kernel_shape[0]
            kernel_width=kernel_shape[1]
            out_height=output_shape[2]
            out_width=output_shape[3]
            stride=strides[0]
            layer_config_dict["layer_name"]=layer_name
            layer_config_dict["in_channel"] =in_channel
            layer_config_dict["in_height"] =in_height
            layer_config_dict["in_width"] =in_width
            layer_config_dict["pad_top"] =pad_top
            layer_config_dict["pad_bottom"] =pad_bottom
            layer_config_dict["pad_left"] =pad_left
            layer_config_dict["pad_right"] =pad_right
            layer_config_dict["kernel_num"] =kernel_num
            layer_config_dict["kernel_height"] =kernel_height
            layer_config_dict["kernel_width"] =kernel_width
            layer_config_dict["out_height"] =out_height
            layer_config_dict["out_width"] = out_width
            layer_config_dict["stride"] = stride
            layer_list.append(layer_config_dict)
        if operation.op_type== "MaxPool"  and layer_type.lower()=="maxpool":
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
            pad_bottom=pads[2]
            pad_left=pads[1]
            pad_right=pads[3]
            kernel_num=output_shape[1]
            kernel_height=kernel_shape[0]
            kernel_width=kernel_shape[1]
            out_height=output_shape[2]
            out_width=output_shape[3]
            stride=strides[0]
            layer_config_dict["layer_name"]=layer_name
            layer_config_dict["in_channel"] =in_channel
            layer_config_dict["in_height"] =in_height
            layer_config_dict["in_width"] =in_width
            layer_config_dict["pad_top"] =pad_top
            layer_config_dict["pad_bottom"] =pad_bottom
            layer_config_dict["pad_left"] =pad_left
            layer_config_dict["pad_right"] =pad_right
            layer_config_dict["kernel_num"] =kernel_num
            layer_config_dict["kernel_height"] =kernel_height
            layer_config_dict["kernel_width"] =kernel_width
            layer_config_dict["out_height"] =out_height
            layer_config_dict["out_width"] = out_width
            layer_config_dict["stride"] = stride
            layer_config_dict["pool_type"]="MAXPOOL"
            layer_list.append(layer_config_dict)
        if operation.op_type== "AveragePool"  and layer_type.lower()=="avgpool":
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
            pad_bottom=pads[2]
            pad_left=pads[1]
            pad_right=pads[3]
            kernel_num=output_shape[1]
            kernel_height=kernel_shape[0]
            kernel_width=kernel_shape[1]
            out_height=output_shape[2]
            out_width=output_shape[3]
            stride=strides[0]
            layer_config_dict["layer_name"]=layer_name
            layer_config_dict["in_channel"] =in_channel
            layer_config_dict["in_height"] =in_height
            layer_config_dict["in_width"] =in_width
            layer_config_dict["pad_top"] =pad_top
            layer_config_dict["pad_bottom"] =pad_bottom
            layer_config_dict["pad_left"] =pad_left
            layer_config_dict["pad_right"] =pad_right
            layer_config_dict["kernel_num"] =kernel_num
            layer_config_dict["kernel_height"] =kernel_height
            layer_config_dict["kernel_width"] =kernel_width
            layer_config_dict["out_height"] =out_height
            layer_config_dict["out_width"] = out_width
            layer_config_dict["stride"] = stride
            layer_config_dict["pool_type"]="AVGPOOL"
            layer_list.append(layer_config_dict)

    return layer_list

if __name__=="__main__":
    f_name=args.fname

