set(INCLUDEDIR ${CMAKE_SOURCE_DIR}/lib/include)

add_library(${PROJECT_NAME} INTERFACE)

set(INTERFACE_HEADERS
    ${CMAKE_SOURCE_DIR}/lib/include/simple_fftw_base.h
    ${CMAKE_SOURCE_DIR}/lib/include/simple_fftw.h
    ${CMAKE_SOURCE_DIR}/lib/include/simple_ifftw.h
)

find_library(FFTW3LIB fftw3)
if(NOT FFTW3LIB)
    message(FATAL_ERROR "FFTW3 library not found. Install it first.")
endif()

target_link_libraries(${PROJECT_NAME} INTERFACE fftw3)
target_include_directories(${PROJECT_NAME} INTERFACE ${INCLUDEDIR})

install(FILES ${INTERFACE_HEADERS}
        DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})