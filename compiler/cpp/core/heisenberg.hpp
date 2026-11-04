#pragma once

#include "compiler/cpp/core/common.hpp"
#include "operators/cpp/pauli.hpp"

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
using qmbboot::operators::_PAULI_I;
using qmbboot::operators::_PAULI_X;
using qmbboot::operators::_PAULI_Y;
using qmbboot::operators::_PAULI_Z;
using qmbboot::operators::_PAULI_PHASE;
using qmbboot::operators::_PAULI_MUL_PHASE_POWER;

enum class HeisenbergWardCharge { hamil, spin_x, spin_y, spin_z };

template <std::size_t L_>
class HeisenbergCompilerCore {
    public:
        static constexpr std::size_t L = L_;
        using PauliString = qmbboot::operators::PauliString<L>;
        static constexpr std::size_t n_words = PauliString::n_words;

    private:
        // the 6 permutations of (X, Y, Z), the internal maps of sym_canon
        static constexpr std::array<std::array<int, 3>, 6> pauli_perms{{
            {{_PAULI_X, _PAULI_Y, _PAULI_Z}},
            {{_PAULI_X, _PAULI_Z, _PAULI_Y}},
            {{_PAULI_Y, _PAULI_X, _PAULI_Z}},
            {{_PAULI_Y, _PAULI_Z, _PAULI_X}},
            {{_PAULI_Z, _PAULI_X, _PAULI_Y}},
            {{_PAULI_Z, _PAULI_Y, _PAULI_X}},
        }};

        // a term coeff * H_t of the Hamiltonian; H_t is a two-site Pauli string P_i P_j with the same P on both sites.
        // words: the mask of H_t
        // coeff: the coefficient of the total H (L times the one of the energy density)
        struct HamilTerm {
            Words<n_words> words;
            Complex coeff;
        };

        // an entry of m_hamil_lookup[site]: the term t touching the site, its other site, and the code of P
        struct HamilLookupEntry {
            std::size_t term;
            std::size_t neighbor;
            int axis;
        };

        std::vector<HamilTerm> m_hamil_terms;
        // m_hamil_lookup[site] lists the Hamiltonian terms touching the site.
        std::vector<std::vector<HamilLookupEntry>> m_hamil_lookup;
        std::vector<PauliString> m_vars;
        VarsLookup<PauliString> m_vars_lookup;

    public:
        // hamil_paulis and hamil_coeffs are the terms of the *total* H (L times the energy density).
        HeisenbergCompilerCore(
            const std::vector<PauliString>& hamil_paulis,
            const std::vector<Complex>& hamil_coeffs
        )
            : m_hamil_lookup(L) {
            if (hamil_paulis.size() != hamil_coeffs.size()) {
                throw std::invalid_argument("hamil_paulis and hamil_coeffs must have the same length");
            }
            for (std::size_t t = 0; t < hamil_paulis.size(); ++t) {
                const auto& words = hamil_paulis[t].words();
                const auto [sites, n_occupied] = hamil_paulis[t].occupied_sites();
                if (n_occupied != 2 || code_of(words, sites[0]) != code_of(words, sites[1])) {
                    throw std::invalid_argument("Hamiltonian terms must be two-site strings P_i P_j");
                }
                const int axis = code_of(words, sites[0]);
                m_hamil_lookup[sites[0]].push_back({t, sites[1], axis});
                m_hamil_lookup[sites[1]].push_back({t, sites[0], axis});
                m_hamil_terms.push_back({words, hamil_coeffs[t]});
            }
        }

        const std::vector<PauliString>& vars() const {
            return m_vars;
        }

        // clear and refill m_vars and m_vars_lookup in place, and return the moments_lookup
        MomentsLookup build_vars(const std::vector<PauliString>& basis) {
            m_vars_lookup.clear();
            m_vars.clear();

            // adjoints[a][r] = (T_r P_a)^dag; Pauli strings are Hermitian.
            std::vector<std::vector<PauliString>> adjoints(basis.size());
            for (std::size_t a = 0; a < basis.size(); ++a) {
                for (std::size_t r = 0; r < L; ++r) {
                    adjoints[a].push_back(basis[a].translate(static_cast<int>(r)).dag());
                }
            }

            auto moment_of = [&](
                std::size_t b, std::size_t a, std::size_t r
            ) -> std::optional<std::pair<PauliString, Complex>> {
                const auto [prod, coeff] = adjoints[a][r].mul(basis[b]);
                const auto canon = canon_of(prod);
                if (!canon) {
                    return std::nullopt;
                }
                return std::pair{*canon, coeff};
            };
            return core::build_vars<PauliString>(basis.size(), L, moment_of, m_vars, m_vars_lookup);
        }

        LinearExprBatch build_ward_identities(HeisenbergWardCharge ward_charge) const {
            switch (ward_charge) {
                case HeisenbergWardCharge::hamil:
                    return core::build_ward_identities<PauliString>(
                        m_vars,
                        [&](const PauliString& var) { return hamil_generators(var); },
                        [&](const PauliString& generator) { return hamil_ward(generator); }
                    );
                case HeisenbergWardCharge::spin_x:
                case HeisenbergWardCharge::spin_y:
                case HeisenbergWardCharge::spin_z: {
                    int axis = _PAULI_X;
                    if (ward_charge == HeisenbergWardCharge::spin_y) {
                        axis = _PAULI_Y;
                    } else if (ward_charge == HeisenbergWardCharge::spin_z) {
                        axis = _PAULI_Z;
                    }
                    return core::build_ward_identities<PauliString>(
                        m_vars,
                        [&](const PauliString& var) { return spin_generators(var, axis); },
                        [&](const PauliString& generator) { return spin_ward(generator, axis); }
                    );
                }
            }
            throw std::invalid_argument("unknown Ward charge");
        }

        // compile the operator sum_t coeffs[t] * paulis[t] into a merged LinearExpr;
        // return std::nullopt if any Pauli string is not an SDP variable
        std::optional<LinearExpr> compile_expr(
            const std::vector<PauliString>& paulis,
            const std::vector<Complex>& coeffs
        ) const {
            if (paulis.size() != coeffs.size()) {
                throw std::invalid_argument("paulis and coeffs must have the same length");
            }
            LinearExpr expr;
            for (std::size_t t = 0; t < paulis.size(); ++t) {
                const auto canon = canon_of(paulis[t]);
                if (!canon) {
                    continue;
                }
                const auto it = m_vars_lookup.find(*canon);
                if (it == m_vars_lookup.end()) {
                    return std::nullopt;
                }
                expr.add(it->second, coeffs[t]);
            }
            expr.merge();
            return expr;
        }

        // calculate [H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) T_s O_\mu(0)
        // as a list of [(O_\mu(0), s, coeff), ...]
        std::vector<std::tuple<PauliString, std::size_t, Complex>> descendants(const PauliString& pstr) const {
            std::vector<std::tuple<PauliString, std::size_t, Complex>> descs;
            for (const auto& [desc, desc_coeff] : hamil_comm(pstr)) {
                const auto desc_rep = desc.trans_canon();

                // desc_rep occupies site 0, so T_s maps desc_rep to desc only if s is an occupied site of desc
                std::size_t shift = 0;
                const auto [sites, n_occupied] = desc.occupied_sites();
                for (std::size_t idx = 0; idx < n_occupied; ++idx) {
                    if (desc_rep.translate(static_cast<int>(sites[idx])) == desc) {
                        shift = sites[idx];
                        break;
                    }
                }

                const auto it = std::find_if(descs.begin(), descs.end(), [&](const auto& entry) {
                    return std::get<1>(entry) == shift && std::get<0>(entry) == desc_rep;
                });
                if (it == descs.end()) {
                    descs.emplace_back(desc_rep, shift, desc_coeff);
                } else if ((std::get<2>(*it) += desc_coeff) == Complex{0.0, 0.0}) {
                    descs.erase(it);
                }
            }
            return descs;
        }

    private:
        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ helpers for bit operations

        // the Pauli code of a site
        static int code_of(const Words<n_words>& x, std::size_t site) {
            const auto bit = 2 * site;
            return static_cast<int>((x[bit / 64] >> (bit % 64)) & 3);
        }

        // x with the Pauli code of a site replaced by code
        static Words<n_words> with_code(Words<n_words> x, std::size_t site, int code) {
            const auto bit = 2 * site;
            x[bit / 64] &= ~(std::uint64_t{3} << (bit % 64));
            x[bit / 64] |= static_cast<std::uint64_t>(code) << (bit % 64);
            return x;
        }

        // call fn once on each non-identity site, in increasing order
        template <class Fn>
        static void for_sites(const Words<n_words>& x, Fn&& fn) {
            std::size_t last = static_cast<std::size_t>(-1);
            x.for_bits([&](std::size_t bit) {
                if (bit / 2 != last) {
                    last = bit / 2;
                    fn(last);
                }
            });
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ symmetry canonicalization

        // quick symmetry filter: SDP variables should have even numbers of
        // X, Y and Z given pi-angle spin-rotation invariant
        static bool sym_allowed(const PauliString& pstr) {
            return pstr.sign_charge() == std::array<int, 3>{0, 0, 0};
        }

        // return the representative Pauli string of its full orbit under symmetry actions of
        // translations, inversion, and permutations of X, Y, Z
        static PauliString sym_canon(const PauliString& pstr) {
            if (pstr == PauliString::identity()) {
                return pstr;
            }

            // The Pauli permutations act within each site and commute with the lattice maps,
            // and the string order compares the highest site first.
            // So among the space group images, i.e. inversion and translations,
            // only those with the lowest highest non-identity site can give the canon (all ties are kept),
            // and the permutations are applied only to these lattice images.
            std::vector<PauliString> lattice_images;
            std::size_t best_top = L;

            for (const bool use_invert : {false, true}) {
                const auto inverted = use_invert ? pstr.invert() : pstr;

                // as in trans_canon, try the translations that move a non-identity site to site 0,
                // and compute the highest non-identity site of each translate.
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
                    lattice_images.push_back(inverted.translate(static_cast<int>(shift)));
                }
            }

            std::optional<PauliString> canon;
            for (const auto& lat_image : lattice_images) {
                for (const auto& perm : pauli_perms) {
                    auto cand = lat_image.permute(perm);
                    if (!canon || cand < *canon) {
                        canon = std::move(cand);
                    }
                }
            }
            return std::move(*canon);
        }

        // return the canonical Pauli string if it can be an SDP variable;
        // return std::nullopt if the expectation value vanishes by symmetry.
        static std::optional<PauliString> canon_of(const PauliString& pstr) {
            if (!sym_allowed(pstr)) {
                return std::nullopt;
            }
            return sym_canon(pstr);
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Hamiltonian dynamics

        // return [H, O] = \sum_t [H_t, O] as a list of (Pauli string, coeff)
        std::vector<std::pair<PauliString, Complex>> hamil_comm(const PauliString& pstr) const {
            const auto& words = pstr.words();
            std::vector<std::pair<PauliString, Complex>> terms;
            std::vector<std::size_t> seen;
            for_sites(words, [&](std::size_t site) {
                const int code = code_of(words, site);
                for (const auto& [t, neighbor, axis] : m_hamil_lookup[site]) {
                    if (std::find(seen.begin(), seen.end(), t) != seen.end()) {
                        continue;
                    }
                    seen.push_back(t);

                    // on a site, P anticommutes with a Pauli operator unless that operator is I or P;
                    // H_t = P_i P_j anticommutes with O if this holds on exactly one of i, j
                    const int neighbor_code = code_of(words, neighbor);
                    const bool anticommute = (code != axis) != (neighbor_code != _PAULI_I && neighbor_code != axis);
                    if (!anticommute) {
                        continue;
                    }
                    const int power = _PAULI_MUL_PHASE_POWER[axis][code] + _PAULI_MUL_PHASE_POWER[axis][neighbor_code];
                    const auto& h = m_hamil_terms[t];
                    terms.emplace_back(PauliString(h.words ^ words), 2.0 * h.coeff * _PAULI_PHASE[power & 3]);
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
        // [H, O] preserves pi-rotation charge, therefore to generate non-trivial stationarity constraints
        // O must have pi-rotation charge ++, i.e. sign charge 000/111.
        // Among them, only 111 sector is K-odd and gives K-even commutators.
        // Given the SDP variables have sign charge 000, candidate generator O must have sign charge 111.
        std::vector<PauliString> hamil_generators(const PauliString& var) const {
            const auto& words = var.words();
            std::vector<PauliString> generators;
            for_sites(words, [&](std::size_t site) {
                const int code = code_of(words, site);
                for (const auto& [t, neighbor, axis] : m_hamil_lookup[site]) {
                    if (code == axis) {
                        continue;
                    }
                    const int neighbor_code = code_of(words, neighbor);
                    if (neighbor_code != _PAULI_I && neighbor_code != axis) {
                        continue;
                    }
                    // trans_canon is cheaper than sym_canon
                    // although it may produce redundant Ward identities
                    generators.push_back(PauliString(words ^ m_hamil_terms[t].words).trans_canon());
                }
            });
            return generators;
        }

        std::optional<LinearExpr> hamil_ward(const PauliString& generator) const {
            const auto& words = generator.words();
            LinearExpr expr;
            std::vector<std::size_t> seen;
            bool missing = false;
            for_sites(words, [&](std::size_t site) {
                if (missing) {
                    // early stop
                    return;
                }
                const int code = code_of(words, site);
                for (const auto& [t, neighbor, axis] : m_hamil_lookup[site]) {
                    if (std::find(seen.begin(), seen.end(), t) != seen.end()) {
                        continue;
                    }
                    seen.push_back(t);

                    // on a site, P anticommutes with a Pauli operator unless that operator is I or P;
                    // H_t = P_i P_j anticommutes with O if this holds on exactly one of i, j
                    const int neighbor_code = code_of(words, neighbor);
                    const bool anticommute = (code != axis) != (neighbor_code != _PAULI_I && neighbor_code != axis);
                    if (!anticommute) {
                        continue;
                    }
                    const auto& h = m_hamil_terms[t];
                    const auto canon = canon_of(PauliString(h.words ^ words));
                    if (!canon) {
                        continue;
                    }
                    const auto it = m_vars_lookup.find(*canon);
                    if (it == m_vars_lookup.end()) {
                        missing = true;
                        return;
                    }

                    const int power = _PAULI_MUL_PHASE_POWER[axis][code] + _PAULI_MUL_PHASE_POWER[axis][neighbor_code];
                    expr.add(it->second, 2.0 * h.coeff * _PAULI_PHASE[power & 3]);
                }
            });
            if (missing) {
                return std::nullopt;
            }
            expr.merge();
            return expr;
        }

        // spin SO(3) Ward identities <[S^a, O]> == 0 with a in x, y, z:
        // O must have sign charge 011, 101 or 110 for a = x, y or z
        // to generate non-trivial Ward identities since SDP variables have sign charge 000.
        std::vector<PauliString> spin_generators(const PauliString& var, int axis) const {
            const auto& words = var.words();
            std::vector<PauliString> generators;
            for_sites(words, [&](std::size_t site) {
                const int code = code_of(words, site);
                if (code == axis) {
                    return;
                }
                // trans_canon is cheaper than sym_canon
                // although it may produce redundant Ward identities
                generators.push_back(PauliString(with_code(words, site, axis ^ code)).trans_canon());
            });
            return generators;
        }

        std::optional<LinearExpr> spin_ward(const PauliString& generator, int axis) const {
            const auto& words = generator.words();
            LinearExpr expr;
            bool missing = false;
            for_sites(words, [&](std::size_t site) {
                if (missing) {
                    // early stop
                    return;
                }
                const int code = code_of(words, site);
                if (code == axis) {
                    return;
                }
                const auto canon = canon_of(PauliString(with_code(words, site, axis ^ code)));
                if (!canon) {
                    return;
                }
                const auto it = m_vars_lookup.find(*canon);
                if (it == m_vars_lookup.end()) {
                    missing = true;
                    return;
                }
                expr.add(it->second, _PAULI_PHASE[_PAULI_MUL_PHASE_POWER[axis][code]]);
            });
            if (missing) {
                return std::nullopt;
            }
            expr.merge();
            return expr;
        }
};

}  // namespace qmbboot::compiler::core
