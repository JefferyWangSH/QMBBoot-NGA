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
    bool initialized = false;
    bool sign_conflict = false;
    MajoranaMonomialSquare<Lx, Ly> canon = MajoranaMonomialSquare<Lx, Ly>::identity();
    int canon_sign = 1;

    for (int up_quarters = 0; up_quarters < 4; ++up_quarters) {
        for (int dn_quarters = 0; dn_quarters < 4; ++dn_quarters) {
            auto [majorana_rotated, majorana_rot_sign] = monomial.majorana_c4_rotate(up_quarters, dn_quarters);

            for (const bool use_exchange : {false, true}) {
                auto exchanged = majorana_rotated;
                auto exchange_sign = majorana_rot_sign;
                if (use_exchange) {
                    auto [next, step_sign] = majorana_rotated.spin_exchange();
                    exchanged = std::move(next);
                    exchange_sign *= step_sign;
                }

                for (int lattice_quarters = 0; lattice_quarters < 4; ++lattice_quarters) {
                    if constexpr (Lx != Ly) {
                        if (lattice_quarters & 1) {
                            continue;
                        }
                    }
                    auto [lattice_rotated, lattice_rot_sign] = exchanged.lattice_c4_rotate(lattice_quarters);
                    lattice_rot_sign *= exchange_sign;

                    for (const bool use_reflect : {false, true}) {
                        auto cand = lattice_rotated;
                        auto cand_sign = lattice_rot_sign;
                        if (use_reflect) {
                            auto [next, step_sign] = lattice_rotated.lattice_reflect_x();
                            cand = std::move(next);
                            cand_sign *= step_sign;
                        }

                        auto [trans_cand, trans_sign] = cand.trans_canon();
                        cand_sign *= trans_sign;

                        if (!initialized || trans_cand.less(canon)) {
                            canon = std::move(trans_cand);
                            canon_sign = cand_sign;
                            sign_conflict = false;
                            initialized = true;
                        } else if (trans_cand == canon && cand_sign != canon_sign) {
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
    return std::pair{std::move(canon), canon_sign};
}

}  // namespace qmbboot::compiler::hubbard_square
