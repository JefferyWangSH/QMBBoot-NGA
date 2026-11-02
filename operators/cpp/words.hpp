#pragma once

#include <array>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace qmbboot::operators {

// A string of bits stored in n_words 64-bit words: bit b is bit b % 64 of word b / 64.
// Words are ordered as the integers sum_w words[w] * 2^(64 w), as Python int masks are,
// so the highest word is compared first (not the lexicographic order of std::array).
template <std::size_t n_words_>
class Words : public std::array<std::uint64_t, n_words_> {
    public:
        static constexpr std::size_t n_words = n_words_;
        using Array = std::array<std::uint64_t, n_words>;

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construction
        Words() : Array{} {}

        Words(const Array& words) : Array(words) {}

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ comparison
        bool operator==(const Words& other) const {
            return static_cast<const Array&>(*this) == static_cast<const Array&>(other);
        }

        std::strong_ordering operator<=>(const Words& other) const {
            for (std::size_t w = n_words; w > 0; --w) {
                if (const auto order = (*this)[w - 1] <=> other[w - 1]; order != 0) {
                    return order;
                }
            }
            return std::strong_ordering::equal;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ bitwise operations
        Words operator^(const Words& other) const {
            Words out;
            for (std::size_t w = 0; w < n_words; ++w) {
                out[w] = (*this)[w] ^ other[w];
            }
            return out;
        }

        Words& operator^=(const Words& other) {
            for (std::size_t w = 0; w < n_words; ++w) {
                (*this)[w] ^= other[w];
            }
            return *this;
        }

        Words operator&(const Words& other) const {
            Words out;
            for (std::size_t w = 0; w < n_words; ++w) {
                out[w] = (*this)[w] & other[w];
            }
            return out;
        }

        Words operator|(const Words& other) const {
            Words out;
            for (std::size_t w = 0; w < n_words; ++w) {
                out[w] = (*this)[w] | other[w];
            }
            return out;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ single bits
        bool get_bit(std::size_t bit) const {
            return ((*this)[bit / 64] >> (bit % 64)) & 1;
        }

        void set_bit(std::size_t bit) {
            (*this)[bit / 64] |= std::uint64_t{1} << (bit % 64);
        }

        void flip_bit(std::size_t bit) {
            (*this)[bit / 64] ^= std::uint64_t{1} << (bit % 64);
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ counting
        // number of set bits
        int popcount() const {
            int count = 0;
            for (std::size_t w = 0; w < n_words; ++w) {
                count += std::popcount((*this)[w]);
            }
            return count;
        }

        // number of set bits shared with other
        int popcount_and(const Words& other) const {
            int count = 0;
            for (std::size_t w = 0; w < n_words; ++w) {
                count += std::popcount((*this)[w] & other[w]);
            }
            return count;
        }

        // number of set bits below bit
        int popcount_below(std::size_t bit) const {
            const std::size_t word = bit / 64;
            const std::size_t offset = bit % 64;
            int count = 0;
            for (std::size_t w = 0; w < word; ++w) {
                count += std::popcount((*this)[w]);
            }
            if (offset) {
                count += std::popcount((*this)[word] & ((std::uint64_t{1} << offset) - 1));
            }
            return count;
        }

        // number of set bits above bit
        int popcount_above(std::size_t bit) const {
            const std::size_t word = bit / 64;
            const std::size_t offset = bit % 64;
            int count = 0;
            if (offset < 63) {
                count += std::popcount((*this)[word] >> (offset + 1));
            }
            for (std::size_t w = word + 1; w < n_words; ++w) {
                count += std::popcount((*this)[w]);
            }
            return count;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ iteration
        // call fn on the set bits in increasing order
        template <class Fn>
        void for_bits(Fn&& fn) const {
            for (std::size_t w = 0; w < n_words; ++w) {
                for (std::uint64_t word = (*this)[w]; word; word &= word - 1) {
                    fn(64 * w + static_cast<std::size_t>(std::countr_zero(word)));
                }
            }
        }
};

}  // namespace qmbboot::operators

template <std::size_t n_words>
struct std::hash<qmbboot::operators::Words<n_words>> {
    std::size_t operator()(const qmbboot::operators::Words<n_words>& words) const {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (const auto word : words) {
            h ^= word + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        h ^= h >> 31;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 29;
        return h;
    }
};
