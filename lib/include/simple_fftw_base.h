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

#ifndef SIMPLE_FFTW_BASE_H
#define SIMPLE_FFTW_BASE_H

#include <concepts>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <memory>
#include <stdexcept>

namespace sfftw {

template <typename T>
concept FFTWType = requires(T a) {
    requires (std::is_same<T, double>::value ||
              std::is_same<T, std::complex<double>>::value);
};

template <FFTWType A, FFTWType B>
class SimpleFFTWBase
{

public:
    SimpleFFTWBase(size_t N): _size(N) {
        in = std::make_unique<A[]>(_size);
        out = std::make_unique<B[]>(_size);
    };

    ~SimpleFFTWBase()
    {
        fftw_destroy_plan(_plan);
    }
    /*
    Get output of latest operation
    */
    std::vector<B> getOutput() const;

    /*
    Get pointer to internal output data.
    */
    B* rawOutput() const noexcept { return out.get(); };
    /*
    Get pointer to internal input data.
    */
    A* rawInput() const noexcept { return in.get(); };

    /*
    Insert data to be used as input for next FFT.
    */
    void input(std::vector<A> &data);

    /*
    Execute FFT operation on input data.
    */
    void execute();

    /*
    Get size of FFT input buffer.
    */
    size_t size() const noexcept { return _size; };

    /*
    Get size of FFT output buffer
    */
    size_t outputSize() const noexcept { return _outputSize; };


protected:
    const size_t _size;
    size_t _outputSize;
    std::unique_ptr<A[]> in;
    std::unique_ptr<B[]> out;
    fftw_plan _plan {nullptr};

    /*
    Set new fftw plan.
    */
    void setPlan(fftw_plan plan) { _plan = plan; };

    /*
    Set size of output data.
    */
   void setOutputSize(size_t size) noexcept { _outputSize = size; };

};

template <FFTWType A, FFTWType B>
void SimpleFFTWBase<A, B>::execute()
{
    fftw_execute(_plan);
}

template <FFTWType A, FFTWType B>
std::vector<B> SimpleFFTWBase<A, B>::getOutput() const
{
    std::vector<B> output(_size);
    output.assign(out.get(), out.get() + _outputSize);
    return output;
}

template <FFTWType A, FFTWType B>
void SimpleFFTWBase<A, B>::input(std::vector<A> &data)
{
    if (data.size() != _size) {
        throw std::length_error("Input data size differs from FFT buffer.");
    }
    for (size_t i = 0; i < _size; ++i) {
        in.get()[i] = data[i];
    }
}

}

#endif