#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstring>
#include <filesystem>
#include <string>
#include <omp.h>

namespace solution {

    std::string compute(const std::string &matrix1_path,
                      const std::string &matrix2_path,
                      int rows, int inner, int cols) {
        
        std::ifstream in_stream1(matrix1_path, std::ios::binary), 
                      in_stream2(matrix2_path, std::ios::binary);
        std::string output_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream out_stream(output_path, std::ios::binary);
        constexpr int TILE_ROW = 16;     
        constexpr int TILE_COL = 48;     
        constexpr int TILE_INNER = 96;  
        size_t bytes_A = static_cast<size_t>(rows) * inner * sizeof(float);
        size_t bytes_B = static_cast<size_t>(inner) * cols * sizeof(float);
        size_t bytes_C = static_cast<size_t>(rows) * cols * sizeof(float);

        float *matrix_A = static_cast<float*>(aligned_alloc(32, bytes_A));
        float *matrix_B = static_cast<float*>(aligned_alloc(32, bytes_B));
        float *matrix_C = static_cast<float*>(aligned_alloc(32, bytes_C));

        in_stream1.read(reinterpret_cast<char*>(matrix_A), bytes_A);
        in_stream2.read(reinterpret_cast<char*>(matrix_B), bytes_B);
        in_stream1.close(); 
        in_stream2.close();

        std::fill_n(matrix_C, static_cast<size_t>(rows) * cols, 0.0f);

        int num_threads = omp_get_num_procs();
        omp_set_num_threads(num_threads);

        #pragma omp parallel
        {
            
            float buffer_A[TILE_ROW * TILE_INNER] __attribute__((aligned(32)));
            float buffer_B[TILE_INNER * TILE_COL] __attribute__((aligned(32)));

            #pragma omp for schedule(dynamic, 1)
            for (int i_start = 0; i_start < rows; i_start += TILE_ROW) {
                int i_end = std::min(rows, i_start + TILE_ROW);
                
                for (int k_start = 0; k_start < inner; k_start += TILE_INNER) {
                    int k_end = std::min(inner,k_start + TILE_INNER);
                    
                    for (int i_idx = i_start; i_idx < i_end; ++i_idx) {
                        std::memcpy(&buffer_A[(i_idx - i_start) * TILE_INNER],
                                    &matrix_A[k_start + static_cast<size_t>(i_idx) * inner],
                                    (k_end - k_start) * sizeof(float));

                    }
                    
                    for (int j_start = 0; j_start < cols; j_start += TILE_COL) {
                        int j_end = std::min(cols,j_start + TILE_COL);
                        
                        for (int k_idx = k_start; k_idx < k_end; ++k_idx) {
                            std::memcpy(&buffer_B[(k_idx - k_start) * TILE_COL],
                                        &matrix_B[j_start + static_cast<size_t>(k_idx) * cols],
                                        (j_end - j_start) * sizeof(float));

                        }
                        
                        for (int i_idx = i_start; i_idx < i_end; ++i_idx) {
                            const float* row_A = &buffer_A[(i_idx - i_start) * TILE_INNER];
                            
                            for (int j_idx = j_start; j_idx < j_end; j_idx += 8) {
                                if (j_idx + 8 <= j_end) {
                                    
                                    __m256 sum = _mm256_loadu_ps(&matrix_C[j_idx + static_cast<size_t>(i_idx) * cols]);                                    
                                    for (int k_idx = 0; k_idx < k_end - k_start; ++k_idx) {
                                        
                                        __m256 aValue = _mm256_set1_ps(row_A[k_idx]);
                                        
                                        __m256 bVector = _mm256_loadu_ps(&buffer_B[k_idx * TILE_COL + (j_idx - j_start)]);
                                        
                                        sum = _mm256_fmadd_ps(aValue, bVector, sum);
                                    }
                                    
                                    _mm256_storeu_ps(&matrix_C[j_idx + static_cast<size_t>(i_idx) * cols], sum);
                                }
                                else {
                                    
                                    for (int j_offset = 0; j_idx + j_offset < j_end; ++j_offset) {
                                        float* result_cell = &matrix_C[static_cast<size_t>(i_idx) * cols + j_idx + j_offset];
                                        float cell_sum = *result_cell;
                                        
                                        for (int k_idx = 0; k_idx < k_end - k_start; ++k_idx) {
                                            cell_sum += row_A[k_idx] * 
                                                buffer_B[k_idx * TILE_COL + (j_idx - j_start) + j_offset];
                                        }
                                        
                                        *result_cell = cell_sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        out_stream.write(reinterpret_cast<const char*>(matrix_C), bytes_C);
        out_stream.close();
        
        free(matrix_A);
        free(matrix_B);
        free(matrix_C);
        
        return output_path;
    }
}