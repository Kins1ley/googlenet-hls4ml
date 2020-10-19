from __future__ import print_function
import numpy as np
import os
import shutil
import math
from onnx import ModelProto, GraphProto, NodeProto, TensorProto
from onnx import optimizer, helper, numpy_helper, shape_inference
import argparse

import numpy as np
import onnx
import os
import glob

from onnx import numpy_helper

test_data_dir = 'image_in/test_data_set_0'



parser = argparse.ArgumentParser(description='onnx reader')
parser.add_argument("--fname", type=str, default="googlenet-7.onnx")
args = parser.parse_args()
MAXMULT = 4096

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    print("cleaning dir {}".format(filepath))
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

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
    for x in model.graph.input:
        if operation.input[input_idx] == x.name:
            return [d.dim_value for d in x.type.tensor_type.shape.dim]
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.input[input_idx]), 0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]


def get_output_shape(model, operation, output_idx=0):
    for x in model.graph.output:
        if operation.output[output_idx] == x.name:
            return [d.dim_value for d in x.type.tensor_type.shape.dim]
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.output[output_idx]),
                          0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]


def sanitize_input_name(name):
    return name.replace("/", "_")


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
    layer_config_dict_base = {}
    for operation in model.graph.node:
        # print(operation.op_type)
        if operation.op_type == "Conv":
            layer_config_dict = layer_config_dict_base.copy()
            input_shape = get_input_shape(model, operation)
            output_shape = get_output_shape(model, operation)
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')
            pads = get_onnx_attribute(operation, 'pads')
            layer_config_dict["in_channel"] = input_shape[1]
            layer_config_dict["in_height"] = input_shape[2]
            layer_config_dict["in_width"] = input_shape[3]
            layer_config_dict["pad_top"] = pads[0]
            layer_config_dict["pad_bottom"] = pads[1]
            layer_config_dict["pad_left"] = pads[2]
            layer_config_dict["pad_right"] = pads[3]
            layer_config_dict["kernel_num"] = output_shape[1]
            layer_config_dict["kernel_height"] = kernel_shape[0]
            layer_config_dict["kernel_width"] = kernel_shape[1]
            layer_config_dict["out_channel"] = output_shape[1]
            layer_config_dict["out_height"] = output_shape[2]
            layer_config_dict["out_width"] = output_shape[3]
            layer_config_dict["stride"] = strides[0]
            layer_config_dict["kernel_channel_DDR_offset"] = 0
            layer_config_dict["bias_DDR_offset"] = 0
            layer_config_dict["out_feature_DDR_offset"] = 0
            layer_config_dict["layer_name"] = "{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                                                              kernel_height=kernel_shape[0],
                                                              kernel_width=kernel_shape[1],
                                                              stride=strides[0],
                                                              input=sanitize_input_name(operation.input[0]),
                                                              output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_type"] = "CONV{kernel_height}x{kernel_width}_S{stride}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["global_weight_name"] = "global_weight_{kernel_height}x{kernel_width}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["global_weight_type"] = "WEIGHT_GLOBAL_{kernel_height}x{kernel_width}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["pe_name"] = "conv_output_reuse{kernel_height}x{kernel_width}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_input_name"] = "{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                                                                     kernel_height=kernel_shape[0],
                                                                     kernel_width=kernel_shape[1],
                                                                     stride=strides[0],
                                                                     input=sanitize_input_name(operation.input[0]),
                                                                     output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_output_name"] = "{output}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["DDR_weight_type"] = "DDR_weight_{kernel_height}x{kernel_width}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["DDR_in_feature"] = "DDR_feature_0"
            layer_config_dict["DDR_out_feature"] = "DDR_feature_1"
            all_layer_config.append(layer_config_dict)
        if operation.op_type == "MaxPool" or operation.op_type == "AveragePool":
            layer_config_dict = layer_config_dict_base.copy()
            input_shape = get_input_shape(model, operation)
            output_shape = get_output_shape(model, operation)
            strides = get_onnx_attribute(operation, 'strides')
            kernel_shape = get_onnx_attribute(operation, 'kernel_shape')
            pads = get_onnx_attribute(operation, 'pads')
            layer_config_dict["in_channel"] = input_shape[1]
            layer_config_dict["in_height"] = input_shape[2]
            layer_config_dict["in_width"] = input_shape[3]
            layer_config_dict["pad_top"] = pads[0]
            layer_config_dict["pad_bottom"] = pads[1]
            layer_config_dict["pad_left"] = pads[2]
            layer_config_dict["pad_right"] = pads[3]
            layer_config_dict["kernel_num"] = output_shape[1]
            layer_config_dict["kernel_height"] = kernel_shape[0]
            layer_config_dict["kernel_width"] = kernel_shape[1]
            layer_config_dict["out_channel"] = output_shape[1]
            layer_config_dict["out_height"] = output_shape[2]
            layer_config_dict["out_width"] = output_shape[3]
            layer_config_dict["stride"] = strides[0]
            layer_config_dict["kernel_channel_DDR_offset"] = 0
            layer_config_dict["bias_DDR_offset"] = 0
            layer_config_dict["out_feature_DDR_offset"] = 0
            layer_config_dict["layer_name"] = "{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                                                              kernel_height=kernel_shape[0],
                                                              kernel_width=kernel_shape[1],
                                                              stride=strides[0],
                                                              input=sanitize_input_name(operation.input[0]),
                                                              output=sanitize_input_name(operation.output[0]))
            if operation.op_type == "MaxPool":
                layer_config_dict["layer_type"] = "MAXPOOL{kernel_height}x{kernel_width}_S{stride}".format(
                    name=sanitize_input_name(operation.output[0])[:-2],
                    kernel_height=kernel_shape[0],
                    kernel_width=kernel_shape[1],
                    stride=strides[0],
                    input=sanitize_input_name(operation.input[0]),
                    output=sanitize_input_name(operation.output[0]))
            if operation.op_type == "AveragePool":
                layer_config_dict["layer_type"] = "AVGPOOL{kernel_height}x{kernel_width}_S{stride}".format(
                    name=sanitize_input_name(operation.output[0])[:-2],
                    kernel_height=kernel_shape[0],
                    kernel_width=kernel_shape[1],
                    stride=strides[0],
                    input=sanitize_input_name(operation.input[0]),
                    output=sanitize_input_name(operation.output[0]))

            layer_config_dict["pe_name"] = "pool{kernel_height}x{kernel_width}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_input_name"] = "{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                                                                     kernel_height=kernel_shape[0],
                                                                     kernel_width=kernel_shape[1],
                                                                     stride=strides[0],
                                                                     input=sanitize_input_name(operation.input[0]),
                                                                     output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_output_name"] = "{output}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                kernel_height=kernel_shape[0],
                kernel_width=kernel_shape[1],
                stride=strides[0],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["DDR_in_feature"] = "DDR_feature_0"
            layer_config_dict["DDR_out_feature"] = "DDR_feature_1"
            all_layer_config.append(layer_config_dict)
        if operation.op_type == "LRN":
            layer_config_dict = layer_config_dict_base.copy()
            input_shape = get_input_shape(model, operation)
            output_shape = get_output_shape(model, operation)
            layer_name = sanitize_input_name(operation.output[0])[:-2]
            depth_radius = get_onnx_attribute(operation, "size")
            layer_config_dict["layer_name"] = layer_name
            layer_config_dict["in_channel"] = input_shape[1]
            layer_config_dict["in_height"] = input_shape[2]
            layer_config_dict["in_width"] = input_shape[3]
            layer_config_dict["out_channel"] = output_shape[1]
            layer_config_dict["out_height"] = output_shape[2]
            layer_config_dict["out_width"] = output_shape[3]

            layer_config_dict["depth_radius"] = depth_radius
            layer_config_dict["layer_name"] = "{name}".format(name=sanitize_input_name(operation.output[0])[:-2],
                                                              input=sanitize_input_name(operation.input[0]),
                                                              output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_type"] = "LRN"
            layer_config_dict["pe_name"] = "LRN"
            layer_config_dict["layer_input_name"] = "{input}".format(name=sanitize_input_name(operation.output[0])[:-2],
                                                                     input=sanitize_input_name(operation.input[0]),
                                                                     output=sanitize_input_name(operation.output[0]))
            layer_config_dict["layer_output_name"] = "{output}".format(
                name=sanitize_input_name(operation.output[0])[:-2],
                input=sanitize_input_name(operation.input[0]),
                output=sanitize_input_name(operation.output[0]))
            layer_config_dict["DDR_in_feature"] = "DDR_feature_0"
            layer_config_dict["DDR_out_feature"] = "DDR_feature_1"
            print(layer_config_dict)
            all_layer_config.append(layer_config_dict)

    for operation in model.graph.node:
        if operation.op_type == "Relu":
            # replace the output name of layers before relu
            inputs = [sanitize_input_name(i) for i in operation.input]
            outputs = [sanitize_input_name(i) for i in operation.output]
            for i in range(len(all_layer_config)):
                for key in all_layer_config[i].keys():
                    if "layer_output" in key:
                        # if all_layer_config[i][key] in inputs:
                        for input_name in inputs:
                            if all_layer_config[i][key][:-2] == input_name[:-2]:
                                # print("(in relu)replacing {} with {}".format(all_layer_config[i][key],outputs[0]))
                                all_layer_config[i][key] = outputs[0]
    for operation in model.graph.node:
        if operation.op_type == "Concat":
            # replace the output name of concatenated layers
            inputs = [sanitize_input_name(i) for i in operation.input]
            outputs = [sanitize_input_name(i) for i in operation.output]
            for i in range(len(all_layer_config)):
                for key in all_layer_config[i].keys():
                    if "layer_output" in key:
                        # if all_layer_config[i][key] in inputs:
                        for input_name in inputs:
                            if all_layer_config[i][key][:-2] == input_name[:-2]:
                                # print("(in concat)replacing {} with {}".format(all_layer_config[i][key],outputs[0]))
                                all_layer_config[i][key] = outputs[0]

    # process last layer
    layer_config_dict = layer_config_dict_base.copy()
    input_shape = [1, 1024, 1, 1]
    pads = [0, 0, 0, 0]
    output_shape = [1, 1000, 1, 1]
    strides = [1, 1]
    kernel_shape = [1, 1]
    layer_config_dict["in_channel"] = input_shape[1]
    layer_config_dict["in_height"] = input_shape[2]
    layer_config_dict["in_width"] = input_shape[3]
    layer_config_dict["pad_top"] = pads[0]
    layer_config_dict["pad_bottom"] = pads[1]
    layer_config_dict["pad_left"] = pads[2]
    layer_config_dict["pad_right"] = pads[3]
    layer_config_dict["kernel_num"] = output_shape[1]
    layer_config_dict["kernel_height"] = kernel_shape[0]
    layer_config_dict["kernel_width"] = kernel_shape[1]
    layer_config_dict["out_channel"] = output_shape[1]
    layer_config_dict["out_height"] = output_shape[2]
    layer_config_dict["out_width"] = output_shape[3]
    layer_config_dict["stride"] = strides[0]
    layer_config_dict["kernel_channel_DDR_offset"] = 0
    layer_config_dict["bias_DDR_offset"] = 0
    layer_config_dict["out_feature_DDR_offset"] = 0
    layer_config_dict["layer_name"] = "loss3_classifier"
    layer_config_dict["layer_type"] = "CONV1x1_S1"
    layer_config_dict["global_weight_name"] = "global_weight_1x1"
    layer_config_dict["global_weight_type"] = "WEIGHT_GLOBAL_1x1"
    layer_config_dict["pe_name"] = "conv_output_reuse1x1"
    layer_config_dict["layer_input_name"] = "pool5_7x7_s1_1"
    layer_config_dict["layer_output_name"] = "out"
    layer_config_dict["DDR_weight_type"] = "DDR_weight_1x1"
    layer_config_dict["DDR_in_feature"] = "DDR_feature_0"
    layer_config_dict["DDR_out_feature"] = "DDR_feature_1"
    all_layer_config.append(layer_config_dict)
    return all_layer_config


def read_value(file_name,all_layer_config,write_to_files=True):
    """
    read weight and bias
    :param file_name:
    :param all_layer_config:
    :param write_to_files:
    :return:
    """
    model = ModelProto()
    weights_dir="./weights"
    with open(file_name, 'rb') as fid:
        model.ParseFromString(fid.read())
    passes = ['fuse_transpose_into_gemm', 'fuse_matmul_add_bias_into_gemm', 'eliminate_nop_transpose',
              'fuse_consecutive_transposes']
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    model = optimizer.optimize(model, passes)
    model = shape_inference.infer_shapes(model)  # have to infer shapes before optimizing the model
    # print(model)
    layer_names = []
    for operation in model.graph.node:
        # print(operation.op_type)
        if operation.op_type == "Conv":
            layer_name = operation.output[0]
            layer_names.append(layer_name[:-2])

    Weight_1x1_all = []
    Weight_3x3_all = []
    Weight_5x5_all = []
    Weight_7x7_all = []
    Bias_all = []
    weight_7x7_channels = 0
    weight_5x5_channels = 0
    weight_3x3_channels = 0
    weight_1x1_channels = 0
    bias_length=0
    initializers = model.graph.initializer

    for layer_config_idx in range(len(all_layer_config)):
        layer_name=all_layer_config[layer_config_idx]["layer_name"]
        if "CONV" in all_layer_config[layer_config_idx]["layer_type"]:
            weight_name = layer_name + "_w_0"
            bias_name = layer_name + "_b_0"
            #print(weight_name,bias_name)
            #bias
            bias, = [t for t in initializers if sanitize_input_name(t.name) == bias_name]
            all_layer_config[layer_config_idx]["bias_DDR_offset"] = bias_length
            bias_length += all_layer_config[layer_config_idx]["out_channel"]
            Bias_all.append(numpy_helper.to_array(bias).flatten().tolist())
            #weight
            if "7x7" in weight_name:
                weight_7x7, = [t for t in initializers if sanitize_input_name(t.name) == weight_name]
                all_layer_config[layer_config_idx]["kernel_channel_DDR_offset"]=weight_7x7_channels
                weight_7x7_channels+=all_layer_config[layer_config_idx]["in_channel"]*all_layer_config[layer_config_idx]["out_channel"]
                Weight_7x7_all.append(numpy_helper.to_array(weight_7x7).flatten().tolist())
            elif "5x5" in weight_name and "reduce" not in weight_name:
                weight_5x5, = [t for t in initializers if sanitize_input_name(t.name) == weight_name]
                all_layer_config[layer_config_idx]["kernel_channel_DDR_offset"]=weight_5x5_channels
                weight_5x5_channels+=all_layer_config[layer_config_idx]["in_channel"]*all_layer_config[layer_config_idx]["out_channel"]
                Weight_5x5_all.append(numpy_helper.to_array(weight_5x5).flatten().tolist())
            elif "3x3" in weight_name and "reduce" not in weight_name:
                weight_3x3, = [t for t in initializers if sanitize_input_name(t.name) == weight_name]
                all_layer_config[layer_config_idx]["kernel_channel_DDR_offset"]=weight_3x3_channels
                weight_3x3_channels+=all_layer_config[layer_config_idx]["in_channel"]*all_layer_config[layer_config_idx]["out_channel"]
                Weight_3x3_all.append(numpy_helper.to_array(weight_3x3).flatten().tolist())
            else:
                weight_1x1, = [t for t in initializers if sanitize_input_name(t.name) == weight_name]
                all_layer_config[layer_config_idx]["kernel_channel_DDR_offset"]=weight_1x1_channels
                weight_1x1_channels+=all_layer_config[layer_config_idx]["in_channel"]*all_layer_config[layer_config_idx]["out_channel"]
                Weight_1x1_all.append(numpy_helper.to_array(weight_1x1).flatten().tolist())
    # print("weight7x7 channels:{}, numbers:{}\n"
    #       "weight5x5 channels:{}, numbers:{}\n"
    #       "weight3x3 channels:{}, numbers:{}\n"
    #       "weight1x1 channels:{}, numbers:{}\n"
    #       "bias: numbers:{}".format(weight_7x7_channels,weight_7x7_channels*7*7,
    #                              weight_5x5_channels,weight_5x5_channels*5*5,
    #                              weight_3x3_channels,weight_3x3_channels*3*3,
    #                              weight_1x1_channels,weight_1x1_channels*1*1,bias_length))
    Weight = [Weight_7x7_all, Weight_5x5_all, Weight_3x3_all, Weight_1x1_all,Bias_all]
    file_names = ["weight_7x7.txt", "weight_5x5.txt", "weight_3x3.txt", "weight_1x1.txt","bias.txt"]
    if write_to_files:
        setDir(weights_dir)
        print("writing weights and input")
        # Load inputs
        inputs = []
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        for i in range(inputs_num):
            input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))
        # Load reference outputs
        ref_outputs = []
        ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
        for i in range(ref_outputs_num):
            output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(output_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            ref_outputs.append(numpy_helper.to_array(tensor))
        print("label:{}".format(np.argmax(ref_outputs[0][0])))
        with open(os.path.join(weights_dir, "image_in.txt"), 'w') as q:
            for i in range(3):
                for j in range(224):
                    for k in range(224):
                        if i+j+k == 0 :
                            q.write(str(inputs[0][0][i][j][k]))
                        else:
                            q.write(',' + str(inputs[0][0][i][j][k]))

        for index in range(len(Weight)):
            #print(file_names[index])
            with open(os.path.join(weights_dir,file_names[index]), 'w') as q:
                for i in range(len(Weight[index])):
                    #print(i)
                    for j in range(len(Weight[index][i])):
                        if i==0 and j==0:
                            q.write(str(Weight[index][i][j]))
                        else:
                            q.write(','+str(Weight[index][i][j]))




def allocate_DRAM_feature(all_layer_config):
    """
    depend on the order of layers in the dict
    :param all_layer_config:
    :return:
    """
    DDR_feature_list = ["DDR_feature_0", "DDR_feature_1", "DDR_feature_2"]
    DDR_feature_idle_status = ["idle", "idle", "idle"]

    def get_idle_DDR_feature():
        idle_index = DDR_feature_idle_status.index("idle")
        return idle_index

    def should_deprecate_feature(feature_name):
        for layer_config_idx in range(len(all_layer_config)):
            if all_layer_config[layer_config_idx]["allocate_output_DRAM"] == False:
                if all_layer_config[layer_config_idx]["layer_input_name"] == feature_name:
                    return False
        return True

    # init
    feature_list = []
    for layer_config_idx in range(len(all_layer_config)):
        # print("layer_input_name:{layer_input_name},layer_output_name:{layer_output_name}".format(**all_layer_config[layer_config_idx]))
        all_layer_config[layer_config_idx]["allocate_output_DRAM"] = False
        if all_layer_config[layer_config_idx]["layer_output_name"] not in feature_list:
            feature_list.append(all_layer_config[layer_config_idx]["layer_output_name"])
    feature_status_dict = {}
    feature_length_dict = {}
    for feature in feature_list:
        # not_ever_exist, deprecated, in_use
        feature_status_dict[feature] = "not_ever_exist"
        feature_length_dict[feature] = 0
    # start allocating
    for layer_config_idx in range(len(all_layer_config)):
        # print("processing layer:{layer_name}, input:{layer_input_name}, output:{layer_output_name}".format(
        #     **all_layer_config[layer_config_idx]))
        all_layer_config[layer_config_idx]["allocate_output_DRAM"] = True
        if feature_status_dict[all_layer_config[layer_config_idx]["layer_output_name"]] == "not_ever_exist":
            feature_status_dict[all_layer_config[layer_config_idx]["layer_output_name"]] = "in_use"
            # get idle DDR feature
            idle_DDR_feature_idx = get_idle_DDR_feature()
            # set DRAM status
            if layer_config_idx != 0:
                all_layer_config[layer_config_idx]["DDR_in_feature"] = DDR_feature_list[
                    DDR_feature_idle_status.index(all_layer_config[layer_config_idx]["layer_input_name"])]
            else:
                all_layer_config[layer_config_idx]["DDR_in_feature"] = "data_0"
            all_layer_config[layer_config_idx]["DDR_out_feature"] = DDR_feature_list[idle_DDR_feature_idx]
            DDR_feature_idle_status[DDR_feature_list.index(all_layer_config[layer_config_idx]["DDR_out_feature"])] = \
            all_layer_config[layer_config_idx]["layer_output_name"]
            print("allocated output feature for layer {layer_name} in DRAM:{DDR_out_feature}".format(**all_layer_config[layer_config_idx]))
        all_layer_config[layer_config_idx]["out_feature_DDR_offset"] = feature_length_dict[
            all_layer_config[layer_config_idx]["layer_output_name"]]
        feature_length_dict[all_layer_config[layer_config_idx]["layer_output_name"]] += \
        all_layer_config[layer_config_idx]["out_channel"] * \
        all_layer_config[layer_config_idx]["out_height"] * \
        all_layer_config[layer_config_idx]["out_width"]
        if should_deprecate_feature(all_layer_config[layer_config_idx]["layer_input_name"]):
            # free the DRAM space for deprecated feature
            if layer_config_idx != 0:
                feature_status_dict[all_layer_config[layer_config_idx]["layer_input_name"]] = "deprecated"
                DDR_feature_idle_status[
                    DDR_feature_idle_status.index(all_layer_config[layer_config_idx]["layer_input_name"])] = "idle"
    max_length=0
    for key in feature_length_dict.keys():
        max_length=max(feature_length_dict[key],max_length)
    print("required shortest feature length:",max_length)
    # validate the result of allocation
    for layer_config_idx in range(len(all_layer_config)):
        feature_DRAM_dict = {}
        # print(
        #     "layer:{layer_name}, input:{layer_input_name}, DRAM:{DDR_in_feature}, output:{layer_output_name}, DRAM:{DDR_out_feature}, output_start_idx:{out_feature_DDR_offset}".format(
        #         **all_layer_config[layer_config_idx]))
        if all_layer_config[layer_config_idx]["layer_input_name"] not in feature_DRAM_dict.keys():
            feature_DRAM_dict[all_layer_config[layer_config_idx]["layer_input_name"]] = \
            all_layer_config[layer_config_idx]["DDR_in_feature"]
        else:
            assert feature_DRAM_dict[all_layer_config[layer_config_idx]["layer_input_name"]] == \
                   all_layer_config[layer_config_idx]["DDR_in_feature"]
        if all_layer_config[layer_config_idx]["layer_output_name"] not in feature_DRAM_dict.keys():
            feature_DRAM_dict[all_layer_config[layer_config_idx]["layer_output_name"]] = \
            all_layer_config[layer_config_idx]["DDR_out_feature"]
        else:
            assert feature_DRAM_dict[all_layer_config[layer_config_idx]["layer_output_name"]] == \
                   all_layer_config[layer_config_idx]["DDR_out_feature"]


if __name__ == "__main__":
    f_name = args.fname
    result = read_onnx(file_name=f_name)
    #allocate_DRAM_feature(result)
    read_value(f_name,result)
