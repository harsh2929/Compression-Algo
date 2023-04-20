#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <lzma.h>

global void lzma_compress_kernel(uint16_t* input, std::size_t Size_input, uint16_t* output, std::size_t output_size, lzma_preset preset, lzma_check check) {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_easy_encoder(&strm, preset, check);
    if (ret != LZMA_OK) {
        printf("Error opening %s\n\n", lzma_strerror(ret));
        return;
    }
    strm.next_in = input + blockIdx.x * blockDim.x;
    strm.avail_in = blockDim.x;
    strm.next_out = output + blockIdx.x * output_size;
    strm.avail_out = output_size;
    ret = lzma_code(&strm, LZMA_RUN);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
        printf("LZMA compression failed: %s\n", lzma_strerror(ret));
        return;
    }
    lzma_end(&strm);
}
#ending lMa
int main(int argc, char* argv[]) {
   
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " input_file" << std::endl;
        return 1;
    }
    std::ifstream input_file(argv[1], std::ios::binary);
    if (!input_file) {
        std::cerr << "Cannot open input file: " << argv[1] << std::endl;
        return 1;
    }
    input_file.seekg(0, std::ios::end);
    std::size_t Size_input = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    char* input_buffer = new char[Size_input];
    input_file.read(input_buffer, Size_input);
    uint16_t* d_input;
    uint16_t* d_output;
    cudaMalloc(&d_input, Size_input);
    cudaMalloc(&d_output, lzma_stream_buffer_bound(Size_input));
    cudaMemcpy(d_input, input_buffer, Size_input, cudaMemcpyHostToDevice);
    std::size_t block_size = 1024;
    std::size_t num_blocks = (Size_input + block_size - 1) / block_size;
    auto start_time = std::chrono::high_resolution_clock::now();
    lzma_compress_kernel<<<num_blocks, 1>>>(d_input, Size_input, d_output, lzma_stream_buffer_bound(Size_input), LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
        return 1;
    }

    std::size_t comp_size = lzma_stream_buffer_bound(Size_input);
    
}