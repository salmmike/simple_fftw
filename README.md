# Simple FFTW

Simple FFTW is a C++20 wrapper around the [fftw3 fast fourier transform library](https://www.fftw.org/).
Currently, the library is built for linux and tested on Ubuntu 22.04.

## Building
The Simple FFTW library is built using CMake. It requires the fftw development libraries.

## Usage
The SimpleFFTW is a template class that supports using either real data as doubles or complex data as std::complex<double>.

Example:

```cpp
#include <simple_fftw.h>
...
size_t N = 2048;
sfftw::SimpleFFTW<double> simpleFFTW(N);

std::vector<double> myData(N);
...

std::vector<std::complex<double>> data = simpleFFTW.fft(myData);

```

