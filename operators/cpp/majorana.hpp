#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmbboot::operators {

template <std::size_t L_>
class MajoranaMonomial {
    static_assert(L_ > 0, "L must be positive");

    public:
        static constexpr std::size_t L = L_;
        static constexpr std::size_t n_modes = 4 * L;
        static constexpr std::size_t n_words = (n_modes + 63) / 64;

    private:
        std::array<std::uint64_t, n_words> m_words{};

    public:
        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construction
        MajoranaMonomial() = default;

        explicit MajoranaMonomial(const std::array<std::uint64_t, n_words>& words) : m_words(words) {
            check_last_word();
        }

        explicit MajoranaMonomial(const std::vector<std::uint64_t>& words) {
            if (words.size() > n_words) {
                for (std::size_t idx = n_words; idx < words.size(); ++idx) {
                    if (words[idx] != 0) {
                        throw std::invalid_argument("word vector exceeds 4L Majorana modes");
                    }
                }
            }
            for (std::size_t idx = 0; idx < std::min(words.size(), n_words); ++idx) {
                m_words[idx] = words[idx];
            }
            check_last_word();
        }

        const std::array<std::uint64_t, n_words>& words() const {
            return m_words;
        }

        static MajoranaMonomial identity() {
            return MajoranaMonomial();
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ comparison
        bool operator==(const MajoranaMonomial& other) const {
            return m_words == other.m_words;
        }

        bool less(const MajoranaMonomial& other) const {
            for (std::size_t idx = n_words; idx > 0; --idx) {
                if (m_words[idx - 1] < other.m_words[idx - 1]) {
                    return true;
                }
                if (m_words[idx - 1] > other.m_words[idx - 1]) {
                    return false;
                }
            }
            return false;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ parsing and formatting
        static std::pair<MajoranaMonomial, int> from_string(const std::string& s) {
            std::istringstream input(s);
            std::string token;
            if (!(input >> token)) {
                return {identity(), 1};
            }
            if (token == "I") {
                std::string extra;
                if (!(input >> extra)) {
                    return {identity(), 1};
                }
                throw std::invalid_argument("invalid Majorana token: " + token);
            }

            MajoranaMonomial monomial;
            int total_sign = 1;
            while (true) {
                if (token.size() < 3) {
                    throw std::invalid_argument("invalid Majorana token: " + token);
                }

                const auto site_str = token.substr(0, token.size() - 2);
                const auto spin_ch = token[token.size() - 2];
                const auto pm_ch = token[token.size() - 1];
                if (site_str.empty() || !std::all_of(site_str.begin(), site_str.end(), [](unsigned char c) { return std::isdigit(c); })) {
                    throw std::invalid_argument("invalid site in Majorana token: " + token);
                }
                if (spin_ch != 'u' && spin_ch != 'd') {
                    throw std::invalid_argument("invalid spin in Majorana token: " + token);
                }
                if (pm_ch != '+' && pm_ch != '-') {
                    throw std::invalid_argument("invalid +/- label in Majorana token: " + token);
                }

                const auto site = static_cast<std::size_t>(std::stoull(site_str));
                if (site >= L) {
                    throw std::invalid_argument("mask exceeds the available 4L Majorana modes");
                }
                const std::size_t spin = spin_ch == 'u' ? 0 : 1;
                const std::size_t pm = pm_ch == '+' ? 0 : 1;
                const auto mode = 4 * site + 2 * spin + pm;
                const auto word_idx = mode / 64;
                const auto bit = mode % 64;

                int count = 0;
                if (bit < 63) {
                    count += std::popcount(monomial.m_words[word_idx] >> (bit + 1));
                }
                for (std::size_t idx = word_idx + 1; idx < n_words; ++idx) {
                    count += std::popcount(monomial.m_words[idx]);
                }
                if (count & 1) {
                    total_sign = -total_sign;
                }
                monomial.m_words[word_idx] ^= std::uint64_t{1} << bit;

                if (!(input >> token)) {
                    break;
                }
            }

            return {monomial, total_sign};
        }

        std::string to_string() const {
            std::ostringstream out;
            bool first = true;
            for (std::size_t mode = 0; mode < n_modes; ++mode) {
                if (!get_bit(mode)) {
                    continue;
                }
                const auto site = mode / 4;
                const auto rem = mode % 4;
                const auto spin = rem / 2;
                const auto pm = rem % 2;
                if (!first) {
                    out << ' ';
                }
                first = false;
                out << site << (spin == 0 ? 'u' : 'd') << (pm == 0 ? '+' : '-');
            }
            return first ? "I" : out.str();
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ physical properties
        int degree() const {
            int total = 0;
            for (const auto word : m_words) {
                total += std::popcount(word);
            }
            return total;
        }

        int dag_phase() const {
            const auto deg = degree();
            return (((deg * (deg - 1)) / 2) & 1) ? -1 : 1;
        }

        std::complex<double> hermitian_phase() const {
            return dag_phase() == 1 ? std::complex<double>{1.0, 0.0} : std::complex<double>{0.0, 1.0};
        }

        std::pair<int, int> fermion_parity() const {
            int up = 0;
            int dn = 0;
            for (std::size_t word_idx = 0; word_idx < n_words; ++word_idx) {
                auto word = m_words[word_idx];
                while (word != 0) {
                    const auto bit = std::countr_zero(word);
                    const auto mode = 64 * word_idx + bit;
                    if (mode % 4 < 2) {
                        ++up;
                    } else {
                        ++dn;
                    }
                    word &= word - 1;
                }
            }
            return {up & 1, dn & 1};
        }

        int k_parity(bool hermitian = true) const {
            int count = 0;
            for (std::size_t mode = 1; mode < n_modes; mode += 2) {
                if (get_bit(mode)) {
                    ++count;
                }
            }
            if (hermitian && dag_phase() == -1) {
                ++count;
            }
            return count & 1;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ physical/symmetry operations
        std::pair<MajoranaMonomial, int> mul(const MajoranaMonomial& other) const {
            int sign = 1;
            for (std::size_t word_idx = 0; word_idx < n_words; ++word_idx) {
                auto word = other.m_words[word_idx];
                while (word != 0) {
                    const auto bit = std::countr_zero(word);

                    int count = 0;
                    if (bit < 63) {
                        count += std::popcount(m_words[word_idx] >> (bit + 1));
                    }
                    for (std::size_t idx = word_idx + 1; idx < n_words; ++idx) {
                        count += std::popcount(m_words[idx]);
                    }
                    if (count & 1) {
                        sign = -sign;
                    }

                    word &= word - 1;
                }
            }

            MajoranaMonomial result;
            for (std::size_t idx = 0; idx < n_words; ++idx) {
                result.m_words[idx] = m_words[idx] ^ other.m_words[idx];
            }
            return {result, sign};
        }

        std::pair<MajoranaMonomial, int> translate(int shift) const {
            std::size_t shift_mod;
            if (shift >= 0) {
                shift_mod = static_cast<std::size_t>(shift) % L;
            } else {
                const auto abs_shift = static_cast<std::size_t>(-(shift + 1)) + 1;
                shift_mod = (L - abs_shift % L) % L;
            }
            if (shift_mod == 0) {
                return {*this, 1};
            }

            MajoranaMonomial result;
            std::size_t swaps = 0;
            for (std::size_t word_idx = 0; word_idx < n_words; ++word_idx) {
                auto word = m_words[word_idx];
                while (word != 0) {
                    const auto bit = std::countr_zero(word);
                    const auto mode = 64 * word_idx + bit;
                    const auto site = mode / 4;
                    const auto rem = mode % 4;
                    const auto new_mode = 4 * ((site + shift_mod) % L) + rem;
                    const auto new_word_idx = new_mode / 64;
                    const auto new_bit = new_mode % 64;

                    int count = 0;
                    if (new_bit < 63) {
                        count += std::popcount(result.m_words[new_word_idx] >> (new_bit + 1));
                    }
                    for (std::size_t idx = new_word_idx + 1; idx < n_words; ++idx) {
                        count += std::popcount(result.m_words[idx]);
                    }
                    swaps += count;
                    result.set_bit(new_mode);
                    word &= word - 1;
                }
            }
            return {result, (swaps & 1) ? -1 : 1};
        }

        std::pair<MajoranaMonomial, int> trans_canon() const {
            auto canon = *this;
            int canon_sign = 1;
            for (std::size_t shift = 1; shift < L; ++shift) {
                auto [cand, cand_sign] = translate(static_cast<int>(shift));
                if (cand.less(canon)) {
                    canon = std::move(cand);
                    canon_sign = cand_sign;
                }
            }
            return {canon, canon_sign};
        }

        std::pair<std::size_t, int> period() const {
            for (std::size_t shift = 1; shift < L; ++shift) {
                auto [cand, cand_sign] = translate(static_cast<int>(shift));
                if (cand == *this) {
                    return {shift, cand_sign};
                }
            }
            return {L, 1};
        }

        std::pair<MajoranaMonomial, int> invert() const {
            MajoranaMonomial result;
            std::size_t swaps = 0;
            for (std::size_t word_idx = 0; word_idx < n_words; ++word_idx) {
                auto word = m_words[word_idx];
                while (word != 0) {
                    const auto bit = std::countr_zero(word);
                    const auto mode = 64 * word_idx + bit;
                    const auto site = mode / 4;
                    const auto rem = mode % 4;
                    const auto new_mode = 4 * ((L - site) % L) + rem;
                    const auto new_word_idx = new_mode / 64;
                    const auto new_bit = new_mode % 64;

                    int count = 0;
                    if (new_bit < 63) {
                        count += std::popcount(result.m_words[new_word_idx] >> (new_bit + 1));
                    }
                    for (std::size_t idx = new_word_idx + 1; idx < n_words; ++idx) {
                        count += std::popcount(result.m_words[idx]);
                    }
                    swaps += count;
                    result.set_bit(new_mode);
                    word &= word - 1;
                }
            }
            return {result, (swaps & 1) ? -1 : 1};
        }

        std::pair<MajoranaMonomial, int> c4_rotate(int up_quarters = 0, int dn_quarters = 0) const {
            static constexpr std::pair<std::size_t, int> c4_rot[4][2] = {
                {{0, +1}, {1, +1}},
                {{1, +1}, {0, -1}},
                {{0, -1}, {1, -1}},
                {{1, -1}, {0, +1}},
            };

            int quarters[2] = {up_quarters % 4, dn_quarters % 4};
            if (quarters[0] < 0) {
                quarters[0] += 4;
            }
            if (quarters[1] < 0) {
                quarters[1] += 4;
            }

            MajoranaMonomial result;
            std::size_t swaps = 0;
            int sign = 1;
            for (std::size_t word_idx = 0; word_idx < n_words; ++word_idx) {
                auto word = m_words[word_idx];
                while (word != 0) {
                    const auto bit = std::countr_zero(word);
                    const auto mode = 64 * word_idx + bit;
                    const auto site = mode / 4;
                    const auto rem = mode % 4;
                    const auto spin = rem / 2;
                    const auto pm = rem % 2;
                    const auto [new_pm, local_sign] = c4_rot[quarters[spin]][pm];
                    const auto new_mode = 4 * site + 2 * spin + new_pm;
                    const auto new_word_idx = new_mode / 64;
                    const auto new_bit = new_mode % 64;

                    int count = 0;
                    if (new_bit < 63) {
                        count += std::popcount(result.m_words[new_word_idx] >> (new_bit + 1));
                    }
                    for (std::size_t idx = new_word_idx + 1; idx < n_words; ++idx) {
                        count += std::popcount(result.m_words[idx]);
                    }
                    swaps += count;
                    sign *= local_sign;
                    result.set_bit(new_mode);
                    word &= word - 1;
                }
            }
            return {result, sign * ((swaps & 1) ? -1 : 1)};
        }

        std::pair<MajoranaMonomial, int> spin_exchange() const {
            MajoranaMonomial result;
            std::size_t swaps = 0;
            for (std::size_t word_idx = 0; word_idx < n_words; ++word_idx) {
                auto word = m_words[word_idx];
                while (word != 0) {
                    const auto bit = std::countr_zero(word);
                    const auto mode = 64 * word_idx + bit;
                    const auto site = mode / 4;
                    const auto rem = mode % 4;
                    const auto spin = rem / 2;
                    const auto pm = rem % 2;
                    const auto new_mode = 4 * site + 2 * (1 - spin) + pm;
                    const auto new_word_idx = new_mode / 64;
                    const auto new_bit = new_mode % 64;

                    int count = 0;
                    if (new_bit < 63) {
                        count += std::popcount(result.m_words[new_word_idx] >> (new_bit + 1));
                    }
                    for (std::size_t idx = new_word_idx + 1; idx < n_words; ++idx) {
                        count += std::popcount(result.m_words[idx]);
                    }
                    swaps += count;
                    result.set_bit(new_mode);
                    word &= word - 1;
                }
            }
            return {result, (swaps & 1) ? -1 : 1};
        }

    private:
        void check_last_word() const {
            const auto rem = n_modes % 64;
            const auto last_mask = rem == 0 ? ~std::uint64_t{0} : ((std::uint64_t{1} << rem) - 1);
            if ((m_words.back() & ~last_mask) != 0) {
                throw std::invalid_argument("word vector exceeds 4L Majorana modes");
            }
        }

        void check_mode(std::size_t mode) const {
            if (mode >= n_modes) {
                throw std::out_of_range("Majorana mode index out of range");
            }
        }

        bool get_bit(std::size_t mode) const {
            check_mode(mode);
            return (m_words[mode / 64] >> (mode % 64)) & 1;
        }

        void set_bit(std::size_t mode) {
            check_mode(mode);
            m_words[mode / 64] |= std::uint64_t{1} << (mode % 64);
        }
};

}  // namespace qmbboot::operators
