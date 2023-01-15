include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(test_simple_fftw test_simple_fftw.cpp)

target_link_libraries(test_simple_fftw
    PRIVATE
    GTest::GTest
    ${PROJECT_NAME})

add_test(testSimpleFFTW test_simple_fftw)
