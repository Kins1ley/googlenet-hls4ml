# GoogLeNet generator
Generate HLS code for GoogLeNet based on the template HLS code

## Code Structure
    - generate_googlenet
        - write all layers of GoogLeNet and corresponding template configurations to corresponding files.
    - generate_conv, generate_lrn, generate_pool
        - generate the code and template configuration of separate layers
    - onnx_info
        - extract information of GoogLeNet from onnx files
    - template_code
        - basic HLS code template, including some typical NN layers
    - origin_code
        - the HLS code of a single nn layer used to do debugging
    - func_templates
        - HLS templates generated from code in "origin_code" of nn layers to generate GoogLeNet
    - out_code
        - generated HLS code for GoogLeNet
## HLS code
- top function
    - googlenet in googlenet.h
- number of PEs and number of BRAM allocated
    - allocate_config.h
    