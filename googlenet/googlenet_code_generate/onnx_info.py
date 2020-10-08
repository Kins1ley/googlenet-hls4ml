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

def sanitize_input_name(name):
    return name.replace("/","_")

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


if __name__=="__main__":
    f_name=args.fname
    result=read_onnx(file_name=f_name)
    print(len(result))
    for i in result:
        print(i)