#pragma once

#include "operators/cpp/majorana.hpp"

#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace qmbboot::operators::majorana {

namespace py = pybind11;

template <std::size_t L>
inline std::array<std::uint64_t, MajoranaMonomial<L>::n_words> mask_to_words(const py::int_& mask) {
    if (PyObject_RichCompareBool(mask.ptr(), py::int_(0).ptr(), Py_LT) == 1) {
        throw std::invalid_argument("mask must be non-negative");
    }
    if (mask.attr("bit_length")().cast<std::size_t>() > MajoranaMonomial<L>::n_modes) {
        throw std::invalid_argument("mask exceeds the available 4L Majorana modes");
    }

    std::array<std::uint64_t, MajoranaMonomial<L>::n_words> words{};
    const auto n_bytes = static_cast<Py_ssize_t>(words.size() * sizeof(std::uint64_t));
    const int flags = Py_ASNATIVEBYTES_LITTLE_ENDIAN
        | Py_ASNATIVEBYTES_UNSIGNED_BUFFER
        | Py_ASNATIVEBYTES_REJECT_NEGATIVE;

    const auto required = PyLong_AsNativeBytes(mask.ptr(), words.data(), n_bytes, flags);
    if (required < 0) {
        throw py::error_already_set();
    }
    if (required > n_bytes) {
        throw std::invalid_argument("mask exceeds the available 4L Majorana modes");
    }
    return words;
}

template <std::size_t N>
inline py::int_ words_to_mask(const std::array<std::uint64_t, N>& words) {
    auto mask = PyLong_FromUnsignedNativeBytes(
        words.data(),
        words.size() * sizeof(std::uint64_t),
        Py_ASNATIVEBYTES_LITTLE_ENDIAN
    );
    if (mask == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::int_>(mask);
}

}  // namespace qmbboot::operators::majorana
