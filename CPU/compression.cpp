#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Implementation:" << argv[0] << "Inputdata" << std::endl;
    }
    std::ifstream Inputdata(argv[1], std::ios::binary);
    if (!Inputdata) {
        std::cerr << "File opening error" << argv[1] << std::endl;
    }
    Inputdata.seekg(0, std::ios::end);
    std::size_t input_size = Inputdata.tellg();
    Inputdata.seekg(0, std::ios::beg);
    char* input_buffer = new char[input_size];
    Inputdata.read(input_buffer, input_size);

    // Compress the input data using LZMA
    std::size_t output_size = lzma_stream_buffer_bound(input_size);
    char* outputbf = new char[output_size];
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_easy_encoder(&strm, LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);
    if (ret != LZMA_OK) {
        std::cerr << "Cannot initialize LZMA encoder: " << lzma_strerror(ret) << std::endl;
    }
    strm.next_in = reinterpret_cast<const uint8_t*>(input_buffer);
    strm.avail_in = input_size;
    strm.next_out = reinterpret_cast<uint8_t*>(outputbf);
    strm.avail_out = output_size;
    auto start_time = std::chrono::high_resolution_clock::now();
    ret = lzma_code(&strm, LZMA_FINISH);
    auto end_time = std::chrono::high_resolution_clock::now();
    if (ret != LZMA_STREAM_END) {
        std::cerr << "LZMA compression failed: " << lzma_strerror(ret) << std::endl;
    }
    std::size_t compressed_size = strm.total_out;
    lzma_end(&strm);
    double compression_ratio = static_cast<double>(input_size) / static_cast<double>(compressed_size);
    double compression_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
    std::size_t max_memory_usage = input_size + output_size;
    double quality = 1.0 - static_cast<double>(compressed_size) / static_cast<double>(input_size);
    double error_rate = 0.0; 
//------------------------------Metrics----------------------------------------------------------------
    std::cout << "Size:" << input_size << " bytes" << std::endl;
    std::cout << "Compressed file size: " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << std::endl;
    std::cout << "Compression time: " << compression_time << " seconds" << std::endl;
    std::cout << "Max memory usage: " << max_memory_usage << " bytes" << std::endl;
