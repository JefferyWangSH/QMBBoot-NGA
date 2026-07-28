#pragma once

#include "operators/cpp/majorana_square.hpp"

#include <cstddef>
#include <optional>
#include <utility>

namespace qmbboot::compiler::hubbard_square {

template <std::size_t Lx, std::size_t Ly>
using MajoranaMonomialSquare = qmbboot::operators::MajoranaMonomialSquare<Lx, Ly>;

template <std::size_t Lx, std::size_t Ly>
inline std::optional<std::pair<MajoranaMonomialSquare<Lx, Ly>, int>> sym_canon(
    const MajoranaMonomialSquare<Lx, Ly>& monomial
) {
    for (const auto& [shift, trans_sign] : monomial.trans_stabilizer()) {
        (void)shift;
        if (trans_sign == -1) {
            return std::nullopt;
        }
    }

    std::optional<MajoranaMonomialSquare<Lx, Ly>> canon;
    int canon_sign = 1;
    bool sign_conflict = false;

    for (int up_quarters = 0; up_quarters < 4; ++up_quarters) {
        for (int dn_quarters = 0; dn_quarters < 4; ++dn_quarters) {
            auto [maj_rotated, maj_rotated_sign] = monomial.majorana_c4_rotate(up_quarters, dn_quarters);

            for (const bool use_exchange : {false, true}) {
                auto exchanged = maj_rotated;
                auto exchanged_sign = maj_rotated_sign;
                if (use_exchange) {
                    auto [mapped, step_sign] = maj_rotated.spin_exchange();
                    exchanged = std::move(mapped);
                    exchanged_sign *= step_sign;
                }

                for (int lat_quarters = 0; lat_quarters < 4; ++lat_quarters) {
                    if constexpr (Lx != Ly) {
                        if (lat_quarters & 1) {
                            continue;
                        }
                    }
                    auto [lat_rotated, step_sign] = exchanged.lattice_c4_rotate(lat_quarters);
                    auto lat_rotated_sign = step_sign * exchanged_sign;

                    for (const bool use_reflect : {false, true}) {
                        auto lat_reflected = lat_rotated;
                        auto lat_reflected_sign = lat_rotated_sign;
                        if (use_reflect) {
                            auto [mapped, step_sign] = lat_rotated.lattice_reflect_x();
                            lat_reflected = std::move(mapped);
                            lat_reflected_sign *= step_sign;
                        }

                        auto [cand, step_sign] = lat_reflected.trans_canon();
                        auto cand_sign = lat_reflected_sign * step_sign;
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
    }

    if (sign_conflict) {
        return std::nullopt;
    }
    return std::pair{std::move(*canon), canon_sign};
}

}  // namespace qmbboot::compiler::hubbard_square
