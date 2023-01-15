/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Author: Mike Salmela
 */

#ifndef SIMPLE_FFTW_H
#define SIMPLE_FFTW_H

#include <fftw3.h>
#include <vector>
#include <concepts>
#include <complex>
#include <type_traits>
#include <memory>
#include <stdexcept>

namespace sfftw {

template <typename T>
concept FFTWType = requires(T a) {
    requires (std::is_same<T, double>::value ||
              std::is_same<T, std::complex<double>>::value);
};

template <FFTWType A>
class SimpleFFTW
{
public:

    SimpleFFTW(size_t s): _size(s)
    {
        init();
    };

    ~SimpleFFTW()
    {
        fftw_destroy_plan(plan);
    }

    /*
    Do FFT operation for input data.
    When using this function, no other operation is needed.
    */
    std::vector<std::complex<double>> fft(std::vector<A> data);

    /*
    Get output of latest operation
    */
    std::vector<std::complex<double>> getOutput() const;

    /*
    Get pointer to internal output data.
    */
    std::complex<double>* rawOutput() const noexcept;

    /*
    Get real part of FFT result
    */
    std::vector<double> real() const;

    /*
    Get imaginary part of FFT result
    */
    std::vector<double> img() const;


    /*
    Insert data to be used as input for next FFT.
    */
    void input(std::vector<A> &data);
    template <typename Iter>
    void input(Iter begin, Iter end);

    /*
    Execute FFT operation on input data.
    */
    void execute();

    /*
    Get size of FFT input buffer.
    */
    size_t size() const noexcept { return size; };

    /*
    Get size of FFT output buffer
    */
    size_t outputSize() const noexcept { return _outputSize; };


private:
    void init();

    const size_t _size;
    size_t _outputSize;
    std::unique_ptr<A[]> in;
    std::unique_ptr<std::complex<double>[]> out;
    fftw_plan plan {nullptr};

};

template <FFTWType A>
void SimpleFFTW<A>::init()
{
    in = std::make_unique<A[]>(_size);
    out = std::make_unique<std::complex<double>[]>(_size);

    if (std::is_floating_point<A>::value) {
        _outputSize = _size/2 - 1;
        plan = fftw_plan_dft_r2c_1d(
            _size,
            reinterpret_cast<double*>(in.get()),
            reinterpret_cast<fftw_complex*>(out.get()),
            FFTW_ESTIMATE
        );
    } else {
        _outputSize = _size;
        plan = fftw_plan_dft_1d(
            _size,
            reinterpret_cast<fftw_complex*>(in.get()),
            reinterpret_cast<fftw_complex*>(out.get()),
            FFTW_FORWARD, FFTW_ESTIMATE
        );
    }
}

template <FFTWType A>
std::vector<std::complex<double>> SimpleFFTW<A>::getOutput() const
{
    std::vector<std::complex<double>> output(_size);
    output.assign(out.get(), out.get() + _outputSize);
    return output;
}

template <FFTWType A>
void SimpleFFTW<A>::input(std::vector<A> &data)
{
    if (data.size() != _size) {
        throw std::length_error("Input data size differs from FFT buffer.");
    }
    for (size_t i = 0; i < _size; ++i) {
        in.get()[i] = data[i];
    }
}

template <FFTWType A>
template <typename Iter>
void SimpleFFTW<A>::input(Iter begin, Iter end)
{
    size_t i = 0;
    while (begin != end) {
        in.get()[i] = *begin;
        ++begin;
        ++i;
    }
}

template <FFTWType A>
std::complex<double>* SimpleFFTW<A>::rawOutput() const noexcept
{
    return out.get();
}

template <FFTWType A>
void SimpleFFTW<A>::execute()
{
    fftw_execute(plan);
}

template <FFTWType A>
std::vector<std::complex<double>> SimpleFFTW<A>::fft(std::vector<A> data)
{
    input(data);
    execute();
    return getOutput();
}

template <FFTWType A>
std::vector<double> SimpleFFTW<A>::real() const
{
    std::vector<double> data(_size);

    std::transform(out.get(), out.get() + _outputSize, std::begin(data),
        [](std::complex<double> d) {
            return d.real();
        }
    );

    return data;
}

template <FFTWType A>
std::vector<double> SimpleFFTW<A>::img() const
{
    std::vector<double> data(_size);

    std::transform(out.get(), out.get() + _size, std::begin(data),
        [](std::complex<double> d) {
            return d.imag();
        }
    );

    return data;
}


}

#endif
