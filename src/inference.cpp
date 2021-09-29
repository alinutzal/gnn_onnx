#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>

//#include <torch/torch.h>
//#include <torch/script.h>
//using namespace torch::indexing;

#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "cuda.h"
#include <cuda_fp16.h>
#include "cuda_runtime_api.h"

struct gnnInferParams
{
    int batchSize;              //!< The input height
    int inputW;              //!< The input width
    int outputSize;          //!< The output size
    std::string modelFile; //!< The filename of the weights file
};

class gnnInfer
{
public:
    gnnInfer(const gnnInferParams& params){}
    gnnInfer(){}
    Ort::Session initOnnxSession(Ort::Env& env);
    std::vector<float> infer(Ort::Session& session,std::tuple<std::vector<float>, std::vector<int64_t>>);
    std::vector<const char*> getInputNodes(Ort::Session& session);
    std::vector<const char*> getOutputNodes(Ort::Session& session);
    std::tuple<std::vector<float>, std::vector<int64_t>> processInputGNN();
};

Ort::Session gnnInfer::initOnnxSession(Ort::Env& env){
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
    // session (we also need to include cuda_provider_factory.h above which defines it)
    // #include "cuda_provider_factory.h"
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    printf("Using Onnxruntime C++ API\n");
    const char* model_path = "../../datanmodels/g_model_full.onnx";
    Ort::Session session(env, model_path, session_options);
    return session;
}

 std::vector<const char*> gnnInfer::getInputNodes(Ort::Session& session){
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                     // Otherwise need vector<vector<>>
     
    //printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    //printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    //printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    //printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    //for (int j = 0; j < input_node_dims.size(); j++)
    //  printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
    return input_node_names;    
}

 std::vector<const char*> gnnInfer::getOutputNodes(Ort::Session& session){
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> output_node_dims; 

    //printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    //printf("Input %d : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    //printf("Output %d : type=%d\n", i, type);

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    //printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
    //for (int j = 0; j < output_node_dims.size(); j++)
    //  printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
    return output_node_names;    
}

std::vector<float> gnnInfer::infer(Ort::Session& session,std::tuple<std::vector<float>, std::vector<int64_t>> input_tensor_values){
    //std::cout<<"got before";
    //std::tuple<std::vector<float>, std::vector<int64_t>> input_tensor_values = processInput();
    std::vector<float> nodes = std::get<0>(input_tensor_values);
    std::vector<int64_t> edgeList = std::get<1>(input_tensor_values);
    size_t nodes_tensor_size = nodes.size();
    size_t edges_tensor_size = edgeList.size();

    //for (int i=0; i<input_tensor_size; i++)
    //   std::cout << input_tensor_values[i] << " ";
    //std::cout<<"got after";
     
    //size_t input_tensor_size = input_tensor_values.size();
    std::vector<int64_t> input_node_dims(2); 
    input_node_dims[1] = 3;
    input_node_dims[0] = nodes_tensor_size/input_node_dims[1];
    std::cout<<"Nodes: "<< input_node_dims[0]<<" "<<input_node_dims[1]<<std::endl;

    std::vector<int64_t> input_edge_dims(2); 
    input_edge_dims[0] = 2;
    input_edge_dims[1] = edges_tensor_size/input_edge_dims[0];
    std::cout<<"Edges: "<<input_edge_dims[0]<<" "<<input_edge_dims[1]<<std::endl;
   
    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    //std::vector<const char*> input_node_names(num_input_nodes);
    //input_node_names = {"actual_input_1"};        

    size_t num_output_nodes = session.GetOutputCount();
    //std::vector<const char*> output_node_names(num_output_nodes);
    //output_node_names = {"output1"};    
    //std::vector<int64_t> output_node_dims; 
    
    // create input tensor object from data values
    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, 
    //                                                          input_node_dims.data(), 2);
    //Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, values_x.data(), input_tensor_size, 
    //                                                          input_node_dims.data(), 2);//, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    std::cout<<"Running gnn inference " << std::endl;
    Ort::MemoryInfo memory_info_cuda("Cuda",OrtArenaAllocator,0, OrtMemTypeDefault);
    Ort::Allocator memory_allocator(session, memory_info_cuda);
    void* inputNodes = memory_allocator.Alloc(sizeof(float) * nodes_tensor_size);
    void* inputEdges = memory_allocator.Alloc(sizeof(int64_t) * edges_tensor_size);
    cudaMemcpy(inputNodes, nodes.data(), sizeof(float) * nodes_tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(inputEdges, edgeList.data(), sizeof(int64_t) * edges_tensor_size, cudaMemcpyHostToDevice);
    Ort::Value nodes_tensor = Ort::Value::CreateTensor<float>(memory_allocator.GetInfo(),reinterpret_cast<float*>(inputNodes), nodes_tensor_size, 
                                                              input_node_dims.data(), 2);
    Ort::Value edges_tensor = Ort::Value::CreateTensor<int64_t>(memory_allocator.GetInfo(),reinterpret_cast<int64_t*>(inputEdges), edges_tensor_size,
                                                              input_edge_dims.data(),2);                                                              

    assert(nodes_tensor.IsTensor());
    assert(edges_tensor.IsTensor());
    //Ort::Value input_tensor[] = {std::move(nodes_tensor), std::move(edges_tensor)};
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = getInputNodes(session);
    std::vector<const char*> output_node_names = getOutputNodes(session);
    // score model & input tensor, get back output tensor
    ort_inputs.push_back(std::move(nodes_tensor));
    ort_inputs.push_back(std::move(edges_tensor));
    //std::vector<const char*> input_names = {"nodes", "edge_index"};
    //std::cout<<input_node_names.data()<<std::endl;
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    // Get pointer to output tensor float values

    float* intarr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {intarr, intarr+ input_edge_dims[0]};
    std::cout << "Size " << input_edge_dims[0] << std::endl;
    return output_tensor_values;
}

std::tuple<std::vector<float>, std::vector<int64_t>> gnnInfer::processInputGNN()
{
    //read input data
    auto file_path = "../../datanmodels/in_g_nodes1000.csv";
    std::ifstream f (file_path);   /* open file */
    if (!f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + std::string(file_path)).c_str());
    }
    std::string line;                    /* string to hold each line */
    //std::vector<std::vector<float>> array;      /* vector of vector<float> for 2d array */
    std::vector<float> nodes;      /* vector of vector<float> for 2d array */

    while (getline (f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        //array.push_back (row);          /* add row to array */
        nodes.insert (nodes.end(),row.begin(),row.end());  
    }
    f.close();

    file_path = "../../datanmodels/in_g_edge_list1000.csv";
    std::ifstream in_file_f (file_path);   /* open file */
    if (!in_file_f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + std::string(file_path)).c_str());
    }
    //std::string line;                     /* string to hold each line */
    //std::vector<std::vector<float>> array;      /* vector of vector<float> for 2d array */
    std::vector<int64_t> edgesList;      /* vector of vector<float> for 2d array */

    while (getline (in_file_f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<int64_t> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        edgesList.insert (edgesList.end(),row.begin(),row.end());  
    }
    in_file_f.close();
    
    std::cout << "Nodes array\n";
    //for (auto& val : input_tensor_values)           /* iterate over vals */
    //    std::cout << val << "  ";        /* output value      */
    //std::cout << "\n";                   /* tidy up with '\n' */
    
    //std::vector<float> input_tensor_values(input.begin(), input.end());
    size_t input_tensor_size = nodes.size();
    std::cout << input_tensor_size << "\n";
    //for (int i=0; i<input_tensor_size; i++)
    //   std::cout << input_tensor_values[i] << " ";
    std::vector<int64_t> input_node_dims(2); 
    input_node_dims[0] = input_tensor_size/3;
    input_node_dims[1] = 3;
    std::cout << input_node_dims[0] << " " << input_node_dims[1] <<"\n";

    std::cout << "Edge List array\n";
    //for (auto& val : input_tensor_values)           /* iterate over vals */
    //    std::cout << val << "  ";        /* output value      */
    //std::cout << "\n";                   /* tidy up with '\n' */
    
    //std::vector<float> input_tensor_values(input.begin(), input.end());
    input_tensor_size = edgesList.size();
    std::cout << input_tensor_size << "\n";
    //for (int i=0; i<input_tensor_size; i++)
    //   std::cout << input_tensor_values[i] << " ";
    //std::vector<int64_t> input_node_dims(2); 
    input_node_dims[0] = input_tensor_size/2;
    input_node_dims[1] = 2;
    std::cout << input_node_dims[0] << " " << input_node_dims[1] <<"\n";


    //auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    //torch::Tensor t = torch::tensor(nodes, {torch::kFloat32});
    //std::cout << t << std::endl;
    //torch::Tensor t = torch::from_blob(input_tensor_values, {input_node_dims}, opts).to(torch::kFloat64);

    std::tuple<std::vector<float>, std::vector<int64_t>> input = {nodes, edgesList};
    return input;
}


//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./1_embed [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mlp/, data/mlp/)"
              << std::endl;
    std::cout << "--use_cuda          Run on the GPU." << std::endl;
    std::cout << "--use_cpu          Run on the CPU." << std::endl;    
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    if (argc <=1) {
        std::cout << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;        
    }
    else {
        if(cmdOptionExists(argv, argv+argc, "-h"))
        {
            printHelpInfo();
            return EXIT_SUCCESS;
        }
    }
    
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        std::cout << "GPU " << id << "\n";
        std::cout << " memory: free=" << free << ", total=" << total << "\n";
    }
    
    std::cout << "Building and running a GPU inference engine for GNN" << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    gnnInfer infer; //initializeSampleParams(args));
    
    Ort::Session session = infer.initOnnxSession(env);
    
    cudaMemGetInfo( &free, &total );
    //std::cout << " memory: free=" << free << ", total=" << total << "\n";
    std::vector<int> e_spatial;
    //long unsigned int e_size, batch_size=800000;
    std::cout<<"GNN: "<<std::endl;
    std::tuple<std::vector<float>, std::vector<int64_t>> inputGNN = infer.processInputGNN();
    std::vector<float> output_gnn = infer.infer(session, inputGNN);
    cudaMemGetInfo( &free, &total );
    std::cout << "  memory: free=" << free << ", total=" << total << "\n";
    return 0;
}

