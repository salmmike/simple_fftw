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

#include <simple_fftw_base.h>

#include <type_traits>
#include <stdexcept>

namespace sfftw {

template <FFTWType A>
class SimpleFFTW: protected SimpleFFTWBase<A, std::complex<double>>
{
public:

    SimpleFFTW(size_t s): SimpleFFTWBase<A, std::complex<double>>(s)
    {
        init();
    };

    /*
    Do FFT operation for input data.
    When using this function, no other operation is needed.
    */
    std::vector<std::complex<double>> fft(std::vector<A> data);
    /*
    Get real part of FFT result
    */
    std::vector<double> real() const;

    /*
    Get imaginary part of FFT result
    */
    std::vector<double> img() const;

    template <typename Iter>
    void input(Iter begin, Iter end);

    using SimpleFFTWBase<A, std::complex<double>>::rawOutput;
    using SimpleFFTWBase<A, std::complex<double>>::rawInput;
    using SimpleFFTWBase<A, std::complex<double>>::input;
    using SimpleFFTWBase<A, std::complex<double>>::execute;
    using SimpleFFTWBase<A, std::complex<double>>::size;
    using SimpleFFTWBase<A, std::complex<double>>::outputSize;
    using SimpleFFTWBase<A, std::complex<double>>::getOutput;

protected:
    void init();
    using SimpleFFTWBase<A, std::complex<double>>::setPlan;
    using SimpleFFTWBase<A, std::complex<double>>::setOutputSize;

};

template <FFTWType A>
void SimpleFFTW<A>::init()
{
    if (std::is_floating_point<A>::value) {
        setOutputSize(size()/2 - 1);
        setPlan( fftw_plan_dft_r2c_1d(
                size(),
                reinterpret_cast<double*>(rawInput()),
                reinterpret_cast<fftw_complex*>(rawOutput()),
                FFTW_ESTIMATE
            )
        );
    } else {
        setOutputSize(size());
        setPlan( fftw_plan_dft_1d(
                size(),
                reinterpret_cast<fftw_complex*>(rawInput()),
                reinterpret_cast<fftw_complex*>(rawOutput()),
                FFTW_FORWARD, FFTW_ESTIMATE
            )
        );
    }
}

template <FFTWType A>
template <typename Iter>
void SimpleFFTW<A>::input(Iter begin, Iter end)
{
    size_t i = 0;
    while (begin != end) {
        rawInput()[i] = *begin;
        ++begin;
        ++i;
    }
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
    std::vector<double> data(size());

    std::transform(rawOutput(), rawOutput() + outputSize(), std::begin(data),
        [](std::complex<double> d) {
            return d.real();
        }
    );

    return data;
}

template <FFTWType A>
std::vector<double> SimpleFFTW<A>::img() const
{
    std::vector<double> data(outputSize());

    std::transform(rawOutput(), rawOutput() + outputSize(), std::begin(data),
        [](std::complex<double> d) {
            return d.imag();
        }
    );

    return data;
}


}

#endif
