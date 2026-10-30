#pragma once

#include "operators/cpp/majorana.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace qmbboot::compiler::hubbard {

template <std::size_t L>
using MajoranaMonomial = qmbboot::operators::MajoranaMonomial<L>;

template <std::size_t L>
inline std::optional<std::pair<MajoranaMonomial<L>, int>> sym_canon(const MajoranaMonomial<L>& monomial) {
    if (monomial.degree() == 0) {
        return std::pair{monomial, 1};
    }

    if (monomial.period().second == -1) {
        return std::nullopt;
    }

    // The internal maps (majorana C4 rotations and spin exchange) act within each site
    // and commute with the lattice maps, and the mask order compares the highest site first.
    // So among the space group images, i.e. inversion and translations,
    // only those with the lowest highest occupied site can give the canon (all ties are kept),
    // and the spin maps are applied only to these lattice images.
    std::vector<std::pair<MajoranaMonomial<L>, int>> lattice_images;
    std::size_t best_top = L;

    for (const bool use_invert : {false, true}) {
        auto inverted = monomial;
        int inverted_sign = 1;
        if (use_invert) {
            auto [mapped, step_sign] = monomial.invert();
            inverted = std::move(mapped);
            inverted_sign = step_sign;
        }

        // as in trans_canon, try the translations that move an occupied site to site 0,
        // and compute the highest occupied site of each translate.
        const auto [sites, n_occupied] = inverted.occupied_sites();
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
            auto [translated, step_sign] = inverted.translate(static_cast<int>(shift));
            lattice_images.emplace_back(std::move(translated), inverted_sign * step_sign);
        }
    }

    std::optional<MajoranaMonomial<L>> canon;
    int canon_sign = 1;
    bool sign_conflict = false;

    for (const auto& [lat_image, lat_image_sign] : lattice_images) {
        for (int up_quarters = 0; up_quarters < 4; ++up_quarters) {
            for (int dn_quarters = 0; dn_quarters < 4; ++dn_quarters) {
                auto [maj_rotated, maj_rotated_sign] = lat_image.c4_rotate(up_quarters, dn_quarters);

                for (const bool use_exchange : {false, true}) {
                    auto cand = maj_rotated;
                    auto cand_sign = lat_image_sign * maj_rotated_sign;
                    if (use_exchange) {
                        auto [mapped, step_sign] = maj_rotated.spin_exchange();
                        cand = std::move(mapped);
                        cand_sign *= step_sign;
                    }

                    if (!canon || cand < *canon) {
                        canon = std::move(cand);
                        canon_sign = cand_sign;
                        sign_conflict = false;
                    } else if (cand == *canon && cand_sign != canon_sign) {
                        sign_conflict = true;
                    }
                }
            }
        }
    }

    if (sign_conflict) {
        return std::nullopt;
    }
    return std::pair{std::move(*canon), canon_sign};
}

}  // namespace qmbboot::compiler::hubbard
