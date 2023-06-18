#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <lzma.h>

#define BLOCK_SIZE 1024

__global__ void lzma_compress_kernel(const uint8_t* input, std::size_t input_size, uint8_t* output, std::size_t output_size, lzma_preset preset, lzma_check check) {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_easy_encoder(&strm, preset, check);
    if (ret != LZMA_OK) {
        printf("Error opening LZMA encoder: %s\n\n", lzma_strerror(ret));
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[1], std::ios::binary);
    if (!input_file) {
        std::cerr << "Cannot open input file: " << argv[1] << std::endl;
        return 1;
    }

    input_file.seekg(0, std::ios::end);
    std::size_t input_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    uint8_t* input_buffer = new uint8_t[input_size];
    input_file.read(reinterpret_cast<char*>(input_buffer), input_size);
    input_file.close();

    uint8_t* d_input;
    uint8_t* d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, lzma_stream_buffer_bound(input_size));
    cudaMemcpy(d_input, input_buffer, input_size, cudaMemcpyHostToDevice);

    std::size_t num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start_time = std::chrono::high_resolution_clock::now();
    lzma_compress_kernel<<<num_blocks, 1>>>(d_input, input_size, d_output, lzma_stream_buffer_bound(input_size), LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::size_t output_size = lzma_stream_buffer_bound(input_size);
    uint8_t* compressed_data = new uint8_t[output_size];
    cudaMemcpy(compressed_data, d_output, output_size, cudaMemcpyDeviceToHost);

    double compression_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    std::cout << "Compression time: " << compression_time << " seconds" << std::endl;

    double compression_ratio = static_cast<double>(input_size) / static_cast<double>(output_size);
    std::cout << "Compression ratio: " << compression_ratio << std::endl;

    // Save compressed data to a file
    std::ofstream output_file("compressed.lzma", std::ios::binary);
    if (output_file) {
        output_file.write(reinterpret_cast<char*>(compressed_data), output_size);
        output_file.close();
        std::cout << "Compressed data saved to compressed.lzma" << std::endl;
    } else {
        std::cerr << "Failed to create output file" << std::endl;
    }

    // Cleanup
    delete[] input_buffer;
    delete[] compressed_data;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
