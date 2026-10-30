#pragma once

#include "operators/cpp/majorana_square.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace qmbboot::compiler::hubbard_square {

template <std::size_t Lx, std::size_t Ly>
using MajoranaMonomialSquare = qmbboot::operators::MajoranaMonomialSquare<Lx, Ly>;

template <std::size_t Lx, std::size_t Ly>
inline std::optional<std::pair<MajoranaMonomialSquare<Lx, Ly>, int>> sym_canon(
    const MajoranaMonomialSquare<Lx, Ly>& monomial
) {
    if (monomial.degree() == 0) {
        return std::pair{monomial, 1};
    }

    for (const auto& [shift, trans_sign] : monomial.trans_stabilizer()) {
        (void)shift;
        if (trans_sign == -1) {
            return std::nullopt;
        }
    }

    // The internal maps (majorana C4 rotations and spin exchange) act within each site
    // and commute with the lattice maps, and the mask order compares the highest site first.
    // So among the space group images, i.e. C4 rotations, reflection and translations,
    // only those with the lowest highest occupied site can give the canon (all ties are kept),
    // and the spin maps are applied only to these lattice images.
    std::vector<std::pair<MajoranaMonomialSquare<Lx, Ly>, int>> lattice_images;
    std::size_t best_top = Lx * Ly;

    for (int lat_quarters = 0; lat_quarters < 4; ++lat_quarters) {
        if constexpr (Lx != Ly) {
            if (lat_quarters & 1) {
                continue;
            }
        }
        auto [lat_rotated, lat_rotated_sign] = monomial.lattice_c4_rotate(lat_quarters);

        for (const bool use_reflect : {false, true}) {
            auto lat_reflected = lat_rotated;
            auto lat_reflected_sign = lat_rotated_sign;
            if (use_reflect) {
                auto [mapped, step_sign] = lat_rotated.lattice_reflect_x();
                lat_reflected = std::move(mapped);
                lat_reflected_sign *= step_sign;
            }

            // as in trans_canon, try the translations that move an occupied row to row 0 and an occupied
            // column to column 0, and compute the highest occupied site of each translate.
            const auto [sites, n_occupied] = lat_reflected.occupied_sites();
            const auto [cols, n_cols] = lat_reflected.occupied_cols();
            const auto [rows, n_rows] = lat_reflected.occupied_rows();
            for (std::size_t row_idx = 0; row_idx < n_rows; ++row_idx) {
                const auto shift_y = (Ly - rows[row_idx]) % Ly;
                for (std::size_t col_idx = 0; col_idx < n_cols; ++col_idx) {
                    const auto shift_x = (Lx - cols[col_idx]) % Lx;
                    std::size_t top = 0;
                    for (std::size_t site_idx = 0; site_idx < n_occupied; ++site_idx) {
                        const auto x = (sites[site_idx] % Lx + shift_x) % Lx;
                        const auto y = (sites[site_idx] / Lx + shift_y) % Ly;
                        top = std::max(top, x + Lx * y);
                    }
                    if (top > best_top) {
                        continue;
                    }
                    if (top < best_top) {
                        best_top = top;
                        lattice_images.clear();
                    }
                    auto [translated, step_sign] = lat_reflected.translate(static_cast<int>(shift_x), static_cast<int>(shift_y));
                    lattice_images.emplace_back(std::move(translated), lat_reflected_sign * step_sign);
                }
            }
        }
    }

    std::optional<MajoranaMonomialSquare<Lx, Ly>> canon;
    int canon_sign = 1;
    bool sign_conflict = false;

    for (const auto& [lat_image, lat_image_sign] : lattice_images) {
        for (int up_quarters = 0; up_quarters < 4; ++up_quarters) {
            for (int dn_quarters = 0; dn_quarters < 4; ++dn_quarters) {
                auto [maj_rotated, maj_rotated_sign] = lat_image.majorana_c4_rotate(up_quarters, dn_quarters);

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

}  // namespace qmbboot::compiler::hubbard_square
