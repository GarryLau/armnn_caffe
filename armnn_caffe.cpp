/* 对单张图片做inference */

/* 用法： ./armnn_caffe image_path image_label
   例如： ./armnn_caffe elephant.jpg 2 */

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

#include "armnn_loader.hpp"


// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData)
{
    return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
}

int main(int argc, char** argv)
{
    // Load a test image and its ground truth label
    std::string img_path = argv[1];
    const int gtlabel = std::stoi(argv[2]);
    std::unique_ptr<Image> input = loadImage(img_path, gtlabel);
    if (input == nullptr){return 1;}

    // Import the Caffe model. Note: use CreateNetworkFromTextFile for text files.
    armnnCaffeParser::ICaffeParserPtr parser = armnnCaffeParser::ICaffeParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("model/ResNet-50_inference.caffemodel",
                                                                   { }, // input taken from file if empty
                                                                   { "prob" }); // output node

    // Find the binding points for the input and output nodes
    armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
    armnnCaffeParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("prob");

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::Compute::CpuAcc);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, runtime->GetDeviceSpec());

    // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    // Run a single inference on the test image
    const int category = 10; /* 分类类别数目 */
    std::array<float, category> output;
    armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
                                                 MakeInputTensors(inputBindingInfo, input->m_image.data()),
                                                 MakeOutputTensors(outputBindingInfo, &output[0]));

    // Convert 1-hot output to an integer label and print
    int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "Predicted: " << label << std::endl;
    std::cout << "   Actual: " << input->m_label << std::endl;
    return 0;
}
