project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/ukf.cpp src/main.cpp src/tools.cpp)


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

    include_directories(/usr/local/include)
    include_directories(/usr/local/opt/openssl/include)
    link_directories(/usr/local/lib)
    link_directories(/usr/local/opt/openssl/lib)
    link_directories(/usr/local/Cellar/libuv/1.15.0/lib)

    # uws - I don't want to install this with sudo!
    include_directories(~/Documents/uni.uni/udacity/selfdriving_cars/term_2/projects/_res/uWebSockets/src)
    link_directories(~/Documents/uni.uni/udacity/selfdriving_cars/term_2/projects/_res/uWebSockets/build)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")  


add_executable(UnscentedKF ${sources})

target_link_libraries(UnscentedKF z ssl uv uWS)
