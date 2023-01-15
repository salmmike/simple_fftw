
#include <simple_fftw.h>

#include <gtest/gtest.h>
#include <iostream>
#include <complex>
#include <math.h>
#include <cassert>
#include <fstream>

#define NAME SimpleFFTWTest

const double epsilon = 0.0001;

const std::vector<double> testResReal
{20.82, -1.97886965, 4.55, 1.86, 3.63, 13.67886965, 1.1, 13.67886965, 3.63, 1.86, 4.55, -1.97886965};

/* Check a == b */
bool compF(double a, double b)
{
    return (a > b - epsilon && a < b + epsilon);
}

template <typename T>
void writeToFile(std::string filename, const std::vector<T> &data)
{
    std::ofstream fp;
    fp.open(filename, std::ios::out);
    for (auto d : data) {
        fp << d << ',';
    }
    fp.close();
}

TEST(NAME, complexShort)
{
    std::vector<std::complex<double>> input
    {5.45, -0.54, 1.81, 1.49, 0.48, 3.98, 0.93, 3.98, 0.48, 1.49, 1.81, -0.54};

    sfftw::SimpleFFTW<std::complex<double>> a(input.size());
    a.input(input);
    a.execute();
    auto data = a.real();

    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(compF(data[i], testResReal[i]))
        << data[i] << " != " << testResReal[i];
    }

    auto comp = a.fft(input);

    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(compF(comp[i].real(), testResReal[i]))
        << comp[i].real() << " != " << testResReal[i];
    }
    //writeToFile("short_data_fftw_complex.csv", a.real());

}

TEST(NAME, testSize)
{
    size_t N = 2048;
    sfftw::SimpleFFTW<double> a(N);
    ASSERT_TRUE(a.size() == N);
}

TEST(NAME, realShort)
{
    std::vector<double> input
    {5.45, -0.54, 1.81, 1.49, 0.48, 3.98, 0.93, 3.98, 0.48, 1.49, 1.81, -0.54};

    sfftw::SimpleFFTW<double> a(input.size());
    a.input(input);

    a.execute();
    auto data = a.real();

    for (size_t i = 0; i < input.size()/2 - 1; ++i) {
        ASSERT_TRUE(compF(data[i], testResReal[i]))
        << data[i] << " != " << testResReal[i]
        << " " << i << " " << input.size()/2 - 1;
    }
    //writeToFile("short_data_fftw_real.csv", a.real());
}

TEST(NAME, insertIter)
{
    std::vector<double> input
    {5.45, -0.54, 1.81, 1.49, 0.48, 3.98, 0.93, 3.98, 0.48, 1.49, 1.81, -0.54};

    sfftw::SimpleFFTW<double> a(input.size());
    a.input(input.begin(), input.end());

    a.execute();
    auto data = a.real();

    for (size_t i = 0; i < input.size()/2 - 1; ++i) {
        ASSERT_TRUE(compF(data[i], testResReal[i]))
        << data[i] << " != " << testResReal[i]
        << " " << i << " " << input.size()/2 - 1;
    }


}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}