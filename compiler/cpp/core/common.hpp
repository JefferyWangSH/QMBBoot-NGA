#pragma once

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <parallel/algorithm>

namespace qmbboot::compiler::core {

using Complex = std::complex<double>;

// For each pair (O_a, O_b) we record a list of terms (r, var_idx, coeff), one per translation r:
//
//      (T_r O_a)^\dag O_b = coeff * m[var_idx],   coeff = +-1 or +-i;
//
// an r for which the product gives no variable has no term.
// Only the pairs a >= b (lower triangle) are recorded because M(q) is Hermitian.
//
// This generates a 3d array [b, a, r] with the missing r removed, flattened to 1d (as in CSR):
// the terms of all pairs are concatenated in the order (b, a, r) into three 1d arrays of the same length,
//
//      rs[t], idxs[t], coeffs[t]   (translation, variable index, coefficient),
//
// and the terms of the pair (a, b) are t in [pair_offsets[k], pair_offsets[k] + pair_sizes[k]),
// k = b*n_basis + a, where the pairs a < b have size 0.

// Key is the type of the SDP variables; it needs ==, < and std::hash<Key>.
template <class Key>
using VarsLookup = std::unordered_map<Key, std::int32_t>;

struct MomentsLookup {
    std::vector<std::int64_t> pair_offsets;
    std::vector<std::int64_t> pair_sizes;
    std::vector<std::int32_t> rs;
    std::vector<std::int32_t> idxs;
    std::vector<Complex> coeffs;
};

template <class Key, class MomentFn>
MomentsLookup build_vars(
    std::size_t n_basis,
    std::size_t n_r,
    MomentFn&& moment_of,
    std::vector<Key>& vars,
    VarsLookup<Key>& vars_lookup
) {
    // entry for each operator O_b
    struct Entry {
        std::int32_t a;
        std::int32_t r;
        Key key;
        Complex coeff;
    };

    MomentsLookup moments_lookup;
    moments_lookup.pair_offsets.reserve(n_basis * n_basis);
    moments_lookup.pair_sizes.reserve(n_basis * n_basis);

    #pragma omp parallel for ordered schedule(dynamic, 1)
    for (std::size_t b = 0; b < n_basis; ++b) {
        std::vector<Entry> out;
        for (std::size_t a = b; a < n_basis; ++a) {
            for (std::size_t r = 0; r < n_r; ++r) {
                auto moment = moment_of(b, a, r);
                if (!moment) {
                    continue;
                }
                out.push_back({static_cast<std::int32_t>(a), static_cast<std::int32_t>(r), moment->first, moment->second});
            }
        }

        // vars, vars_lookup and moments_lookup are filled serially,
        // so the variables are numbered in the order (b, a, r) whatever the number of threads.
        // In particular, vars and vars_lookup are modified in place.
        #pragma omp ordered
        {
            auto entry = out.begin();
            for (std::size_t a = 0; a < n_basis; ++a) {
                const std::size_t start = moments_lookup.rs.size();
                moments_lookup.pair_offsets.push_back(static_cast<std::int64_t>(start));
                for (; entry != out.end() && static_cast<std::size_t>(entry->a) == a; ++entry) {
                    const auto [it, inserted] = vars_lookup.try_emplace(entry->key, static_cast<std::int32_t>(vars.size()));
                    if (inserted) {
                        vars.push_back(entry->key);
                    }
                    moments_lookup.rs.push_back(entry->r);
                    moments_lookup.idxs.push_back(it->second);
                    moments_lookup.coeffs.push_back(entry->coeff);
                }
                moments_lookup.pair_sizes.push_back(static_cast<std::int64_t>(moments_lookup.rs.size() - start));
            }
        }
    }
    return moments_lookup;
}

// linear combinations of SDP variables, sum_k coeff_k * m[idx_k], without the constant term
struct LinearExpr {
    std::vector<std::pair<std::int32_t, Complex>> terms;
    bool merged = true;

    void add(std::int32_t idx, Complex coeff) {
        terms.emplace_back(idx, coeff);
        merged = false;
    }

    void merge() {
        if (merged) {
            return;
        }
        std::stable_sort(terms.begin(), terms.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        std::size_t n_kept = 0;
        for (auto term = terms.begin(); term != terms.end();) {
            const auto idx = term->first;
            Complex sum = term->second;
            for (++term; term != terms.end() && term->first == idx; ++term) {
                sum += term->second;
            }
            if (sum != Complex{0.0, 0.0}) {
                terms[n_kept++] = {idx, sum};
            }
        }
        terms.resize(n_kept);
        merged = true;
    }
};

// batch of merged LinearExpr in CSR form
struct LinearExprBatch {
    std::vector<std::int64_t> offsets{0};
    std::vector<std::int32_t> idxs;
    std::vector<Complex> coeffs;

    void append(const LinearExpr& expr) {
        assert(expr.merged);
        for (const auto& [idx, coeff] : expr.terms) {
            idxs.push_back(idx);
            coeffs.push_back(coeff);
        }
        offsets.push_back(static_cast<std::int64_t>(idxs.size()));
    }
};

// exhaust all the Ward identities <[C, O]> = 0 associated with charge C
// that can be compiled on the given SDP variables as a LinearExprBatch.
//
//      vars:                    SDP variables as a vector of Key;
//      generators_of(var):      the canonical generators O of a variable;
//      ward_identity_of(O):     the *merged* expression of the Ward identity,
//                               or nothing if it cannot be compiled.

template <class Key, class GeneratorsFn, class WardIdentityFn>
LinearExprBatch build_ward_identities(
    const std::vector<Key>& vars,
    GeneratorsFn&& generators_of,
    WardIdentityFn&& ward_identity_of
) {
    // variables are taken in chunks to bound the memory of the generators before deduplication
    constexpr std::size_t chunk = 65536;
    std::vector<std::vector<Key>> local(std::min(chunk, vars.size()));
    std::vector<Key> generators;
    for (std::size_t v0 = 0; v0 < vars.size(); v0 += chunk) {
        const std::size_t v1 = std::min(vars.size(), v0 + chunk);

        #pragma omp parallel for schedule(dynamic, 16)
        for (std::size_t v = v0; v < v1; ++v) {
            local[v - v0] = generators_of(vars[v]);
        }

        // flatten the generators
        const std::size_t n_sorted = generators.size();
        std::vector<std::size_t> offsets(v1 - v0 + 1, 0);
        for (std::size_t v = v0; v < v1; ++v) {
            offsets[v - v0 + 1] = offsets[v - v0] + local[v - v0].size();
        }
        generators.resize(n_sorted + offsets.back());
        #pragma omp parallel for schedule(static)
        for (std::size_t v = v0; v < v1; ++v) {
            std::copy(local[v - v0].begin(), local[v - v0].end(), generators.begin() + n_sorted + offsets[v - v0]);
            std::vector<Key>().swap(local[v - v0]);
        }

        // dedupe the generators
        const auto mid = generators.begin() + n_sorted;
        __gnu_parallel::sort(mid, generators.end());
        generators.erase(std::unique(mid, generators.end()), generators.end());
        std::inplace_merge(generators.begin(), generators.begin() + n_sorted, generators.end());
        generators.erase(std::unique(generators.begin(), generators.end()), generators.end());
    }
    std::vector<std::vector<Key>>().swap(local);

    std::vector<std::optional<LinearExpr>> exprs(generators.size());
    #pragma omp parallel for schedule(dynamic, 16)
    for (std::size_t idx = 0; idx < generators.size(); ++idx) {
        exprs[idx] = ward_identity_of(generators[idx]);
    }

    LinearExprBatch batch;
    for (auto& expr : exprs) {
        if (expr && !expr->terms.empty()) {
            batch.append(*expr);
        }
        expr.reset();
    }
    return batch;
}

}  // namespace qmbboot::compiler::core
