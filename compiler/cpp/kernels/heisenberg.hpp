#pragma once

#include "operators/cpp/pauli.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

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

    if (pstr == PauliString<L>::identity()) {
        return pstr;
    }

    // The Pauli permutations act within each site and commute with the lattice maps,
    // and the string order compares the highest site first.
    // So among the space group images, i.e. inversion and translations,
    // only those with the lowest highest non-identity site can give the canon (all ties are kept),
    // and the permutations are applied only to these lattice images.
    std::vector<PauliString<L>> lattice_images;
    std::size_t best_top = L;

    for (const bool use_invert : {false, true}) {
        auto inv_image = use_invert ? pstr.invert() : pstr;

        // as in trans_canon, try the translations that move a non-identity site to site 0,
        // and compute the highest non-identity site of each translate.
        const auto [sites, n_occupied] = inv_image.occupied_sites();
        for (std::size_t idx = 0; idx < n_occupied; ++idx) {
            const auto shift = (L - sites[idx]) % L;
            std::size_t top = 0;
            for (std::size_t site_idx = 0; site_idx < n_occupied; ++site_idx) {
                top = std::max(top, (sites[site_idx] + shift) % L);
            }
            if (top > best_top) {
                continue;
            }
            if (top < best_top) {
                best_top = top;
                lattice_images.clear();
            }
            lattice_images.push_back(inv_image.translate(static_cast<int>(shift)));
        }
    }

    std::optional<PauliString<L>> canon;

    for (const auto& lat_image : lattice_images) {
        for (const auto& perm : perms) {
            auto cand = lat_image.permute(perm);
            if (!canon || cand < *canon) {
                canon = std::move(cand);
            }
        }
    }

    return std::move(*canon);
}

}  // namespace qmbboot::compiler::heisenberg
