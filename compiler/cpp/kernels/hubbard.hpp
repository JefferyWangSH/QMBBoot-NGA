#pragma once

#include "operators/cpp/majorana.hpp"

#include <cstddef>
#include <optional>
#include <utility>

namespace qmbboot::compiler::hubbard {

template <std::size_t L>
using MajoranaMonomial = qmbboot::operators::MajoranaMonomial<L>;

template <std::size_t L>
inline std::optional<std::pair<MajoranaMonomial<L>, int>> sym_canon(const MajoranaMonomial<L>& monomial) {
    if (monomial.period().second == -1) {
        return std::nullopt;
    }

    std::optional<MajoranaMonomial<L>> canon;
    int canon_sign = 1;
    bool sign_conflict = false;

    for (int up_quarters = 0; up_quarters < 4; ++up_quarters) {
        for (int dn_quarters = 0; dn_quarters < 4; ++dn_quarters) {
            auto [rotated, rotated_sign] = monomial.c4_rotate(up_quarters, dn_quarters);

            for (const bool use_exchange : {false, true}) {
                auto exchanged = rotated;
                auto exchanged_sign = rotated_sign;
                if (use_exchange) {
                    auto [mapped, step_sign] = rotated.spin_exchange();
                    exchanged = std::move(mapped);
                    exchanged_sign *= step_sign;
                }

                for (const bool use_invert : {false, true}) {
                    auto inverted = exchanged;
                    auto inverted_sign = exchanged_sign;
                    if (use_invert) {
                        auto [mapped, step_sign] = exchanged.invert();
                        inverted = std::move(mapped);
                        inverted_sign *= step_sign;
                    }

                    auto [cand, step_sign] = inverted.trans_canon();
                    auto cand_sign = inverted_sign * step_sign;
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
