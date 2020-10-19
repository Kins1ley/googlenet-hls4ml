import os
import shutil
top_function_mark=   "/////////////top_function/////////////"
template_config_mark="/////////////template_config/////////////"
allocate_config_mark="/////////////allocate_config/////////////"


layer_config_dict={"left_bracket":"{","right_bracket":"}"}
layer_config_dict["layer_name"]="pool1_3x3_s2"
layer_config_dict["layer_type"]="MAXPOOL3x3_S2"
layer_config_dict["pe_name"]="pool3x3"
layer_config_dict["layer_input_name"]="conv1_7x7_s2_1"
layer_config_dict["layer_output_name"]="pool1_3x3_s2_1"
layer_config_dict["DDR_in_feature"]="DDR_feature_0"
layer_config_dict["DDR_out_feature"]="DDR_feature_1"
def generate_pool_template_from_origin():
    """
    generate the template file
    :return:
    """
    with open("./origin_code/pool_template_origin.h") as f:
        tmp_file=open("./origin_code/pool_template_tmp.h","w")
        while True:
            text = f.readline()
            if not text:
                break
            text = text.replace("{","{{")
            text = text.replace("}","}}")
            text = text.replace("pool1_3x3_s2_1_config", "{layer_output_name}_config")
            text = text.replace("conv1_7x7_s2_1_config", "{layer_input_name}_config")
            text = text.replace("pool1_3x3_s2_1", "{DDR_out_feature}")
            text = text.replace("conv1_7x7_s2_1", "{DDR_in_feature}")
            text = text.replace("pool1_3x3_s2","{layer_name}")
            text = text.replace("MAXPOOL3x3_S2","{layer_type}")
            text = text.replace("pool3x3", "{pe_name}")

            print(text,file=tmp_file,end="")
            #print(text)
        tmp_file.close()
    #os.remove("pool_template.h")
    os.rename("./origin_code/pool_template_tmp.h", "./function_templates/pool_template.h")

def generate_pool(template_name="./function_templates/pool_template.h",part="top_function"):
    """
    generate code according to the template file
    :param template_name:
    :param part: the part to write in the file.
    :return:
    """
    found_top_function=False
    found_template_config=False
    found_allocate_config=False
    code=""
    with open(template_name) as f:
        while True:
            text = f.readline()
            if not text:
                break
            #print(text)
            if top_function_mark in text:
                if not found_top_function:
                    found_top_function=True
                else:
                    found_top_function=False
                continue

            if template_config_mark in text:
                if not found_template_config:
                    found_template_config=True
                else:
                    found_template_config=False
                continue

            if allocate_config_mark in text:
                if not found_allocate_config:
                    found_allocate_config=True
                else:
                    found_allocate_config=False
                continue

            if (part=="top_function" and found_top_function) or \
                (part=="template_config" and found_template_config) or \
                (part=="allocate_config" and found_allocate_config):
                code+=text
    return code

if __name__ == "__main__":
    generate_pool_template_from_origin()
    code=generate_pool()
    #print(code)
    #print(code.format(**layer_config_dict))