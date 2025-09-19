if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(embeding.axera)

# bsp
if(NOT BSP_MSP_DIR)
    # 判断 /soc/lib/libax_engine.so 是否存在，以确定是否为板端编译
    if(EXISTS /soc/lib/libax_engine.so)
        message(STATUS "Detected board,BSP_MSP_DIR = /soc")
        set(BSP_MSP_DIR /soc)
    else()
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
    endif()
endif()
message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")
include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

include_directories(include)

add_library(embeding SHARED
    src/embeding.cpp
    src/tokenizer/tokenizer.cpp
    src/ax_model_runner/ax_model_runner_ax650.cpp
    src/utils/cqdm.cpp
    src/utils/memory_utils.cpp
)


function(build_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} embeding ax_engine ax_interpreter ax_sys)
    install(TARGETS ${test_name}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/${CMAKE_SYSTEM_PROCESSOR}
    )
endfunction()

build_test(test_tokenizer)
build_test(test_tokenizer_v2)
build_test(test_embeding)
build_test(test_api)

install(FILES include/embeding.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(FILES tests/tokenizer.txt
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share
)
install(TARGETS embeding
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SYSTEM_PROCESSOR}
)
