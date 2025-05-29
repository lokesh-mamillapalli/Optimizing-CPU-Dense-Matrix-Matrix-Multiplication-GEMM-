add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(studentlib PRIVATE -O3 -march=native)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)
endif()
