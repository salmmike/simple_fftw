set(INCLUDEDIR ${CMAKE_SOURCE_DIR}/lib/include)

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} INTERFACE fftw3)
target_include_directories(${PROJECT_NAME} INTERFACE ${INCLUDEDIR})
