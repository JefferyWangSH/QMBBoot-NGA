#pragma once

#include "compiler/cpp/core/common.hpp"
#include "operators/cpp/majorana.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace qmbboot::compiler::core {

using qmbboot::operators::Words;
using qmbboot::operators::_MAJORANA_SPIN_LADDER_TERMS;
using qmbboot::operators::_MAJORANA_ETA_LADDER_TERMS;

enum class HubbardWardCharge { hamil, number_up, number_dn, spin_plus, eta_plus };

template <std::size_t L_>
class HubbardCompilerCore {
    public:
        static constexpr std::size_t L = L_;
        using MajoranaMonomial = qmbboot::operators::MajoranaMonomial<L>;
        static constexpr std::size_t n_words = MajoranaMonomial::n_words;

    private:
        // a term coeff * H_t of the Hamiltonian; H_t is a Majorana monomial.
        // words: the mask of H_t (bit m set if gamma_m is in H_t)
        // coeff: the coefficient of the total H (L times the one of the energy density)
        // modes: the modes of H_t in increasing order, kept for the sign of H_t O in normal order
        struct HamilTerm {
            Words<n_words> words;
            Complex coeff;
            std::vector<std::size_t> modes;
        };

        bool m_use_ph_sym;
        std::vector<HamilTerm> m_hamil_terms;
        // m_hamil_lookup[site] lists the indices of Hamiltonian terms touching the site.
        std::vector<std::vector<std::size_t>> m_hamil_lookup;
        std::vector<MajoranaMonomial> m_vars;
        VarsLookup<MajoranaMonomial> m_vars_lookup;

    public:
        // hamil_monomials and hamil_coeffs are the terms of the *total* H (L times the energy density).
        HubbardCompilerCore(
            const std::vector<MajoranaMonomial>& hamil_monomials,
            const std::vector<Complex>& hamil_coeffs,
            bool use_ph_sym
        )
            : m_use_ph_sym(use_ph_sym), m_hamil_lookup(L) {
            if (hamil_monomials.size() != hamil_coeffs.size()) {
                throw std::invalid_argument("hamil_monomials and hamil_coeffs must have the same length");
            }
            for (std::size_t t = 0; t < hamil_monomials.size(); ++t) {
                HamilTerm term{hamil_monomials[t].words(), hamil_coeffs[t], {}};
                term.words.for_bits([&](std::size_t mode) { term.modes.push_back(mode); });
                for_sites(term.words, [&](std::size_t site) { m_hamil_lookup[site].push_back(t); });
                m_hamil_terms.push_back(std::move(term));
            }
        }

        const std::vector<MajoranaMonomial>& vars() const {
            return m_vars;
        }

        // clear and refill m_vars and m_vars_lookup in place, and return the moments_lookup
        MomentsLookup build_vars(const std::vector<MajoranaMonomial>& basis) {
            m_vars_lookup.clear();
            m_vars.clear();

            // adjoints[a][r] = (N[T_r O_a], s) with (T_r O_a)^dag = s * N[T_r O_a].
            std::vector<std::vector<std::pair<MajoranaMonomial, int>>> adjoints(basis.size());
            for (std::size_t a = 0; a < basis.size(); ++a) {
                const int dag_phase = basis[a].dag_phase();
                for (std::size_t r = 0; r < L; ++r) {
                    auto [translated, trans_sign] = basis[a].translate(static_cast<int>(r));
                    adjoints[a].emplace_back(std::move(translated), trans_sign * dag_phase);
                }
            }

            auto moment_of = [&](
                std::size_t b, std::size_t a, std::size_t r
            ) -> std::optional<std::pair<MajoranaMonomial, Complex>> {
                const auto& [monomial_ar, adj_sign] = adjoints[a][r];
                const auto [prod, mul_sign] = monomial_ar.mul(basis[b]);
                const auto canon = canon_of(prod);
                if (!canon) {
                    return std::nullopt;
                }
                const Complex coeff = Complex(adj_sign * mul_sign * canon->second) / prod.hermitian_phase();
                return std::pair{canon->first, coeff};
            };
            return core::build_vars<MajoranaMonomial>(basis.size(), L, moment_of, m_vars, m_vars_lookup);
        }

        LinearExprBatch build_ward_identities(HubbardWardCharge ward_charge) const {
            switch (ward_charge) {
                case HubbardWardCharge::hamil:
                    return core::build_ward_identities<MajoranaMonomial>(
                        m_vars,
                        [&](const MajoranaMonomial& var) { return hamil_generators(var); },
                        [&](const MajoranaMonomial& generator) { return hamil_ward(generator); }
                    );
                case HubbardWardCharge::number_up:
                case HubbardWardCharge::number_dn: {
                    const std::size_t spin = ward_charge == HubbardWardCharge::number_up ? 0 : 1;
                    return core::build_ward_identities<MajoranaMonomial>(
                        m_vars,
                        [&](const MajoranaMonomial& var) { return number_generators(var, spin); },
                        [&](const MajoranaMonomial& generator) { return number_ward(generator, spin); }
                    );
                }
                case HubbardWardCharge::spin_plus:
                    return core::build_ward_identities<MajoranaMonomial>(
                        m_vars,
                        [&](const MajoranaMonomial& var) { return spin_generators(var, 0); },
                        [&](const MajoranaMonomial& generator) { return spin_ward(generator, 0); }
                    );
                case HubbardWardCharge::eta_plus:
                    return core::build_ward_identities<MajoranaMonomial>(
                        m_vars,
                        [&](const MajoranaMonomial& var) { return eta_generators(var, 0); },
                        [&](const MajoranaMonomial& generator) { return eta_ward(generator, 0); }
                    );
            }
            throw std::invalid_argument("unknown Ward charge");
        }

        // compile the operator sum_t coeffs[t] * monomials[t] into a merged LinearExpr;
        // return std::nullopt if any monomial is not an SDP variable
        std::optional<LinearExpr> compile_expr(
            const std::vector<MajoranaMonomial>& monomials,
            const std::vector<Complex>& coeffs
        ) const {
            if (monomials.size() != coeffs.size()) {
                throw std::invalid_argument("monomials and coeffs must have the same length");
            }
            LinearExpr expr;
            for (std::size_t t = 0; t < monomials.size(); ++t) {
                const auto& monomial = monomials[t];
                const auto canon = canon_of(monomial);
                if (!canon) {
                    continue;
                }
                const auto it = m_vars_lookup.find(canon->first);
                if (it == m_vars_lookup.end()) {
                    return std::nullopt;
                }
                expr.add(it->second, coeffs[t] * double(canon->second) / monomial.hermitian_phase());
            }
            expr.merge();
            return expr;
        }

        // calculate [H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) T_s O_\mu(0)
        // as a list of [(O_\mu(0), s, coeff), ...]
        std::vector<std::tuple<MajoranaMonomial, std::size_t, Complex>> descendants(
            const MajoranaMonomial& monomial
        ) const {
            std::vector<std::tuple<MajoranaMonomial, std::size_t, Complex>> descs;
            for (const auto& [desc, desc_coeff] : hamil_comm(monomial)) {
                const auto desc_rep = desc.trans_canon().first;

                // desc_rep occupies site 0, so T_s maps desc_rep to desc only if s is an occupied site of desc
                std::size_t shift = 0;
                int shift_sign = 1;
                const auto [sites, n_sites] = desc.occupied_sites();
                for (std::size_t idx = 0; idx < n_sites; ++idx) {
                    const auto [shifted, trans_sign] = desc_rep.translate(static_cast<int>(sites[idx]));
                    if (shifted == desc) {
                        shift = sites[idx];
                        shift_sign = trans_sign;
                        break;
                    }
                }

                const Complex coeff = desc_coeff * double(shift_sign);
                const auto it = std::find_if(descs.begin(), descs.end(), [&](const auto& entry) {
                    return std::get<1>(entry) == shift && std::get<0>(entry) == desc_rep;
                });
                if (it == descs.end()) {
                    descs.emplace_back(desc_rep, shift, coeff);
                } else if ((std::get<2>(*it) += coeff) == Complex{0.0, 0.0}) {
                    descs.erase(it);
                }
            }
            return descs;
        }

    private:
        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ helpers for bit operations

        // call fn once on each site with an occupied mode, in increasing order
        template <class Fn>
        static void for_sites(const Words<n_words>& x, Fn&& fn) {
            std::size_t last = static_cast<std::size_t>(-1);
            x.for_bits([&](std::size_t mode) {
                if (mode / 4 != last) {
                    last = mode / 4;
                    fn(last);
                }
            });
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ symmetry canonicalization

        // quick symmetry filter for monomials that are symmetry eigenoperators;
        // SDP variables should have even fermion parity for each spin, even K parity,
        // and, at half filling on even L, even particle-hole parity.
        bool sym_allowed(const MajoranaMonomial& monomial) const {
            const auto [up, dn] = monomial.fermion_parity();
            if (up || dn) {
                return false;
            }
            if (monomial.k_parity(true)) {
                return false;
            }
            return !m_use_ph_sym || monomial.ph_parity() == 0;
        }

        // return the representative monomial of its full orbit under symmetry actions
        // (translations, inversion, Majorana C4 rotations of each spin, spin exchange) and its sign;
        // if some action maps the monomial to itself with the opposite sign,
        // its expectation value is zero and std::nullopt is returned.
        static std::optional<std::pair<MajoranaMonomial, int>> sym_canon(const MajoranaMonomial& monomial) {
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
            std::vector<std::pair<MajoranaMonomial, int>> lattice_images;
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

            std::optional<MajoranaMonomial> canon;
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

        // return the canonical monomial and its sign if the monomial can be an SDP variable;
        // return std::nullopt if the expectation value vanishes by symmetry.
        std::optional<std::pair<MajoranaMonomial, int>> canon_of(const MajoranaMonomial& monomial) const {
            if (!sym_allowed(monomial)) {
                return std::nullopt;
            }
            return sym_canon(monomial);
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Hamiltonian dynamics

        // return [H, O] = \sum_t [H_t, O] as a list of (monomial, coeff)
        std::vector<std::pair<MajoranaMonomial, Complex>> hamil_comm(const MajoranaMonomial& monomial) const {
            const auto& words = monomial.words();
            std::vector<std::pair<MajoranaMonomial, Complex>> terms;
            std::vector<std::size_t> seen;
            for_sites(words, [&](std::size_t site) {
                for (const auto t : m_hamil_lookup[site]) {
                    if (std::find(seen.begin(), seen.end(), t) != seen.end()) {
                        continue;
                    }
                    seen.push_back(t);

                    const auto& h = m_hamil_terms[t];
                    if (!(h.words.popcount_and(words) & 1)) {
                        // H_t commutes with O if they share an even number of modes
                        continue;
                    }
                    int parity = 0;
                    for (const auto mode : h.modes) {
                        parity += words.popcount_below(mode);
                    }
                    const double sign = (parity & 1) ? -1.0 : 1.0;
                    terms.emplace_back(MajoranaMonomial(h.words ^ words), 2.0 * h.coeff * sign);
                }
            });
            return terms;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ build Ward identities
        // For each charge C, the Ward identities <[C, O]> = 0 are built
        // as LinearExprBatch in three steps:
        //
        //    *_generators(var): the generators O whose Ward identity may contain var,
        //                       as translation representatives;
        //    *_ward(O):         the *merged* LinearExpr of <[C, O]>,
        //                       or std::nullopt if a term is not an SDP variable.
        //
        // Then pass them to core::build_ward_identities,
        // as defined in compiler/cpp/core/common.hpp, which returns the LinearExprBatch.

        // stationarity Ward identities <[H, O]> == 0:
        // candidate generator O must have odd K parity and even spin-resolved fermion parity.
        std::vector<MajoranaMonomial> hamil_generators(const MajoranaMonomial& var) const {
            const auto& words = var.words();
            std::vector<MajoranaMonomial> generators;
            for_sites(words, [&](std::size_t site) {
                for (const auto t : m_hamil_lookup[site]) {
                    const auto& h = m_hamil_terms[t];
                    if (!(h.words.popcount_and(words) & 1)) {
                        // H_t commutes with var if they share an even number of modes
                        continue;
                    }
                    // trans_canon is cheaper than sym_canon
                    // although it may produce redundant Ward identities
                    generators.push_back(MajoranaMonomial(words ^ h.words).trans_canon().first);
                }
            });
            return generators;
        }

        std::optional<LinearExpr> hamil_ward(const MajoranaMonomial& generator) const {
            const auto& words = generator.words();
            LinearExpr expr;
            std::vector<std::size_t> seen;
            bool missing = false;
            for_sites(words, [&](std::size_t site) {
                if (missing) {
                    // early stop
                    return;
                }
                for (const auto t : m_hamil_lookup[site]) {
                    if (std::find(seen.begin(), seen.end(), t) != seen.end()) {
                        continue;
                    }
                    seen.push_back(t);

                    const auto& h = m_hamil_terms[t];
                    if (!(h.words.popcount_and(words) & 1)) {
                        // H_t commutes with O if they share an even number of modes
                        continue;
                    }
                    const MajoranaMonomial prod(h.words ^ words);
                    const auto canon = canon_of(prod);
                    if (!canon) {
                        continue;
                    }
                    const auto it = m_vars_lookup.find(canon->first);
                    if (it == m_vars_lookup.end()) {
                        missing = true;
                        return;
                    }

                    int parity = 0;
                    for (const auto mode : h.modes) {
                        parity += words.popcount_below(mode);
                    }
                    const double sign = (parity & 1) ? -1.0 : 1.0;
                    expr.add(it->second, 2.0 * h.coeff * sign * double(canon->second) / prod.hermitian_phase());
                }
            });
            if (missing) {
                return std::nullopt;
            }
            expr.merge();
            return expr;
        }

        // charge U(1) Ward identities <[N_s, O]> == 0, s = 0/1 for up/down:
        // candidate generator O must have odd K parity and even spin-resolved fermion parity.
        std::vector<MajoranaMonomial> number_generators(const MajoranaMonomial& var, std::size_t spin) const {
            const auto& words = var.words();
            std::vector<MajoranaMonomial> generators;
            words.for_bits([&](std::size_t mode) {
                const auto site = mode / 4;
                const auto rem = mode % 4;
                if (rem / 2 != spin) {
                    // n_{i,s} commutes with the mode if it is of the other spin
                    return;
                }
                const auto plus_mode = 4 * site + 2 * spin;
                const auto minus_mode = plus_mode + 1;
                if (words.get_bit(plus_mode) == words.get_bit(minus_mode)) {
                    // n_{i,s} commutes with var if var holds both or neither of the two modes
                    return;
                }
                Words<n_words> generator = words;
                generator.flip_bit(plus_mode);
                generator.flip_bit(minus_mode);
                // trans_canon is cheaper than sym_canon
                // although it may produce redundant Ward identities
                generators.push_back(MajoranaMonomial(generator).trans_canon().first);
            });
            return generators;
        }

        std::optional<LinearExpr> number_ward(const MajoranaMonomial& generator, std::size_t spin) const {
            const auto& words = generator.words();
            LinearExpr expr;
            bool missing = false;
            words.for_bits([&](std::size_t mode) {
                if (missing) {
                    // early stop
                    return;
                }
                const auto site = mode / 4;
                const auto rem = mode % 4;
                if (rem / 2 != spin) {
                    // n_{i,s} commutes with the mode if it is of the other spin
                    return;
                }
                const auto plus_mode = 4 * site + 2 * spin;
                const auto minus_mode = plus_mode + 1;
                const bool has_plus = words.get_bit(plus_mode);
                if (has_plus == words.get_bit(minus_mode)) {
                    // n_{i,s} commutes with O if O holds both or neither of the two modes
                    return;
                }

                Words<n_words> prod_words = words;
                prod_words.flip_bit(plus_mode);
                prod_words.flip_bit(minus_mode);
                const MajoranaMonomial prod(prod_words);
                const auto canon = canon_of(prod);
                if (!canon) {
                    return;
                }
                const auto it = m_vars_lookup.find(canon->first);
                if (it == m_vars_lookup.end()) {
                    missing = true;
                    return;
                }

                const double sign = has_plus ? -1.0 : 1.0;
                expr.add(it->second, Complex{0.0, 1.0} * sign * double(canon->second) / prod.hermitian_phase());
            });
            if (missing) {
                return std::nullopt;
            }
            expr.merge();
            return expr;
        }

        // spin SU(2) Ward identities <[S_\pm, O]> == 0, pm = 0/1 for S_+/S_-:
        //
        //      1) <[S_z, O]> == 0 have been covered by <[N_s, O]> == 0.
        //
        //      2) since S_\pm carries spin-resolved parity (1, 1), candidate generator O has
        //         spin-resolved parity (1, 1) so that [S_\pm, O] can land in the even variable sector (0, 0).
        //
        //      3) its K parity is not fixed because S_\pm mixes local K parities.

        std::vector<MajoranaMonomial> spin_generators(const MajoranaMonomial& var, std::size_t pm) const {
            const auto& words = var.words();
            const auto& table = _MAJORANA_SPIN_LADDER_TERMS[pm];
            std::vector<MajoranaMonomial> generators;
            for_sites(words, [&](std::size_t site) {
                for (const auto& [rem1, rem2, coeff] : table) {
                    const auto mode1 = 4 * site + rem1;
                    const auto mode2 = 4 * site + rem2;
                    if (words.get_bit(mode1) == words.get_bit(mode2)) {
                        // commutes with var if var holds both or neither of the two modes
                        continue;
                    }
                    Words<n_words> generator = words;
                    generator.flip_bit(mode1);
                    generator.flip_bit(mode2);
                    // trans_canon is cheaper than sym_canon
                    // although it may produce redundant Ward identities
                    generators.push_back(MajoranaMonomial(generator).trans_canon().first);
                }
            });
            return generators;
        }

        std::optional<LinearExpr> spin_ward(const MajoranaMonomial& generator, std::size_t pm) const {
            const auto& words = generator.words();
            const auto& table = _MAJORANA_SPIN_LADDER_TERMS[pm];
            LinearExpr expr;
            bool missing = false;
            for_sites(words, [&](std::size_t site) {
                if (missing) {
                    // early stop
                    return;
                }
                for (const auto& [rem1, rem2, coeff] : table) {
                    const auto mode1 = 4 * site + rem1;
                    const auto mode2 = 4 * site + rem2;
                    if (words.get_bit(mode1) == words.get_bit(mode2)) {
                        // commutes with O if O holds both or neither of the two modes
                        continue;
                    }

                    Words<n_words> prod_words = words;
                    prod_words.flip_bit(mode1);
                    prod_words.flip_bit(mode2);
                    const MajoranaMonomial prod(prod_words);
                    const auto canon = canon_of(prod);
                    if (!canon) {
                        continue;
                    }
                    const auto it = m_vars_lookup.find(canon->first);
                    if (it == m_vars_lookup.end()) {
                        missing = true;
                        return;
                    }

                    const int swaps = words.popcount_below(mode1) + words.popcount_below(mode2);
                    const double sign = (swaps & 1) ? -1.0 : 1.0;
                    expr.add(it->second, 2.0 * coeff * sign * double(canon->second) / prod.hermitian_phase());
                }
            });
            if (missing) {
                return std::nullopt;
            }
            expr.merge();
            return expr;
        }

        // eta-pairing SU(2) Ward identities <[eta^a, O]> == 0, pm = 0/1 for eta^+/eta^-;
        // <[eta^z, O]> == 0 has been covered by total-number conservation.
        std::vector<MajoranaMonomial> eta_generators(const MajoranaMonomial& var, std::size_t pm) const {
            const auto& words = var.words();
            const auto& table = _MAJORANA_ETA_LADDER_TERMS[pm];
            std::vector<MajoranaMonomial> generators;
            for_sites(words, [&](std::size_t site) {
                for (const auto& [rem1, rem2, coeff] : table) {
                    const auto mode1 = 4 * site + rem1;
                    const auto mode2 = 4 * site + rem2;
                    if (words.get_bit(mode1) == words.get_bit(mode2)) {
                        // commutes with var if var holds both or neither of the two modes
                        continue;
                    }
                    Words<n_words> generator = words;
                    generator.flip_bit(mode1);
                    generator.flip_bit(mode2);
                    // trans_canon is cheaper than sym_canon
                    // although it may produce redundant Ward identities
                    generators.push_back(MajoranaMonomial(generator).trans_canon().first);
                }
            });
            return generators;
        }

        std::optional<LinearExpr> eta_ward(const MajoranaMonomial& generator, std::size_t pm) const {
            const auto& words = generator.words();
            const auto& table = _MAJORANA_ETA_LADDER_TERMS[pm];
            LinearExpr expr;
            bool missing = false;
            for_sites(words, [&](std::size_t site) {
                if (missing) {
                    // early stop
                    return;
                }
                const double stagger = (site & 1) ? -1.0 : 1.0;
                for (const auto& [rem1, rem2, coeff] : table) {
                    const auto mode1 = 4 * site + rem1;
                    const auto mode2 = 4 * site + rem2;
                    if (words.get_bit(mode1) == words.get_bit(mode2)) {
                        // commutes with O if O holds both or neither of the two modes
                        continue;
                    }

                    Words<n_words> prod_words = words;
                    prod_words.flip_bit(mode1);
                    prod_words.flip_bit(mode2);
                    const MajoranaMonomial prod(prod_words);
                    const auto canon = canon_of(prod);
                    if (!canon) {
                        continue;
                    }
                    const auto it = m_vars_lookup.find(canon->first);
                    if (it == m_vars_lookup.end()) {
                        missing = true;
                        return;
                    }

                    const int swaps = words.popcount_below(mode1) + words.popcount_below(mode2);
                    const double sign = (swaps & 1) ? -1.0 : 1.0;
                    expr.add(it->second, 2.0 * stagger * coeff * sign * double(canon->second) / prod.hermitian_phase());
                }
            });
            if (missing) {
                return std::nullopt;
            }
            expr.merge();
            return expr;
        }
};

}  // namespace qmbboot::compiler::core
