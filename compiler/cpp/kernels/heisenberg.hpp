#pragma once

#include "operators/cpp/pauli.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace qmbboot::compiler::heisenberg {

template <std::size_t L>
using PauliString = qmbboot::operators::PauliString<L>;
using qmbboot::operators::_PAULI_X;
using qmbboot::operators::_PAULI_Y;
using qmbboot::operators::_PAULI_Z;

template <std::size_t L>
inline PauliString<L> sym_canon(const PauliString<L>& pstr) {
    static constexpr std::array<std::array<int, 3>, 6> perms{{
        {{_PAULI_X, _PAULI_Y, _PAULI_Z}},
        {{_PAULI_X, _PAULI_Z, _PAULI_Y}},
        {{_PAULI_Y, _PAULI_X, _PAULI_Z}},
        {{_PAULI_Y, _PAULI_Z, _PAULI_X}},
        {{_PAULI_Z, _PAULI_X, _PAULI_Y}},
        {{_PAULI_Z, _PAULI_Y, _PAULI_X}},
    }};

    bool initialized = false;
    PauliString<L> canon = PauliString<L>::identity();

    for (const bool use_invert : {false, true}) {
        auto inv_image = use_invert ? pstr.invert() : pstr;
        for (const auto& perm : perms) {
            auto cand = inv_image.permute(perm).trans_canon();
            if (!initialized || cand.less(canon)) {
                canon = std::move(cand);
                initialized = true;
            }
        }
    }

    return canon;
}

}  // namespace qmbboot::compiler::heisenberg
