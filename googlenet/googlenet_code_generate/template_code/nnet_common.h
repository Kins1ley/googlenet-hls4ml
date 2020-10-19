#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_
#include<fstream>
#include<string>
#include<iostream>
#include<cstdlib>
#include <sstream>
#define WEIGHTS_DIR "weights"
namespace nnet {

template<class weight_type, size_t length,size_t height, size_t width>
void load_weights_from_txt(weight_type w[length][height][width], const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while (std::getline(iss, token, ',')) {
            std::istringstream(token) >> w[i / (height * width)][(i / width) % height][i % width];
            i++;
        }

        if (length * height * width != i) {
            std::cerr << "ERROR: Expected " << length * height * width << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

template<class data_type, size_t length>
void load_data_from_txt(data_type w[length], const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while (std::getline(iss, token, ',')) {
            std::istringstream(token) >> w[i];
            i++;
        }

        if (length!= i) {
            std::cerr << "ERROR: Expected " << length  << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

}
#endif