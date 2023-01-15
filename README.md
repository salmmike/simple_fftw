# Simple FFTW

Simple FFTW is a C++20 wrapper around the [fftw3 fast fourier transform library](https://www.fftw.org/).
Currently, the library is built for linux and tested on Ubuntu 22.04.

## Building
The Simple FFTW library is built using CMake. It requires the fftw development libraries.
The library is tested using ctest and Google tests.
Run ctest after building to run unit tests.

## Usage
The SimpleFFTW is a template class that supports using either real data as doubles or complex data as std::complex<double>.

Inverse FFT can be done with the SimpleIFFTW class. The usage is uniform to the SimpleFFTW.

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

