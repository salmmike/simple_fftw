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

#ifndef SIMPLE_IFFTW_H
#define SIMPLE_IFFTW_H

#include <simple_fftw_base.h>
#include <type_traits>

namespace sfftw {

template <FFTWType A>
class SimpleIFFTW: protected SimpleFFTWBase<std::complex<double>, A>
{
public:

    SimpleIFFTW(size_t s): SimpleFFTWBase<std::complex<double>, A>(s)
    {
        init();
    };

    /*
    Do IFFT operation for input data.
    When using this function, no other operation is needed.
    If normalize == true, the result is divided by size.
    */
    std::vector<A> ifft(std::vector<std::complex<double>> data, bool normalize=false);

    template <typename Iter>
    void input(Iter begin, Iter end);

    using SimpleFFTWBase<std::complex<double>, A>::rawOutput;
    using SimpleFFTWBase<std::complex<double>, A>::rawInput;
    using SimpleFFTWBase<std::complex<double>, A>::input;
    using SimpleFFTWBase<std::complex<double>, A>::execute;
    using SimpleFFTWBase<std::complex<double>, A>::size;
    using SimpleFFTWBase<std::complex<double>, A>::outputSize;
    using SimpleFFTWBase<std::complex<double>, A>::getOutput;

protected:
    void init();
    using SimpleFFTWBase<std::complex<double>, A>::setPlan;
    using SimpleFFTWBase<std::complex<double>, A>::setOutputSize;
};

template <FFTWType A>
void SimpleIFFTW<A>::init()
{
    if (std::is_floating_point<A>::value) {
        setOutputSize(size()/2 - 1);
        setPlan( fftw_plan_dft_c2r_1d(
                size(),
                reinterpret_cast<fftw_complex*>(rawInput()),
                reinterpret_cast<double*>(rawOutput()),
                FFTW_ESTIMATE
            )
        );
    } else {
        setOutputSize(size());
        setPlan( fftw_plan_dft_1d(
                size(),
                reinterpret_cast<fftw_complex*>(rawInput()),
                reinterpret_cast<fftw_complex*>(rawOutput()),
                FFTW_BACKWARD, FFTW_ESTIMATE
            )
        );
    }
}

template <FFTWType A>
template <typename Iter>
void SimpleIFFTW<A>::input(Iter begin, Iter end)
{
    size_t i = 0;
    while (begin != end) {
        rawInput()[i] = *begin;
        ++begin;
        ++i;
    }
}

template <FFTWType A>
std::vector<A> SimpleIFFTW<A>::ifft(std::vector<std::complex<double>> data, bool normalize)
{
    input(data);
    execute();
    if (normalize) {
        std::transform(rawOutput(), rawOutput() + outputSize(), rawOutput(),
            [this](A d) {
                return d / this->size();
            }
        );
    }
    return getOutput();
}

}

#endif
