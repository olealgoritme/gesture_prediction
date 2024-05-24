cmake_minimum_required(VERSION 3.10)
project(GestureClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake/dlib")

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(generate_data src/generate_data.cpp)
add_executable(train src/train.cpp)
add_executable(run src/run.cpp)
add_executable(finetuning src/finetuning.cpp)

target_link_libraries(generate_data dlib::dlib ${OpenCV_LIBS})
target_link_libraries(train dlib::dlib ${OpenCV_LIBS})
target_link_libraries(run dlib::dlib ${OpenCV_LIBS})
target_link_libraries(finetuning dlib::dlib ${OpenCV_LIBS})
