#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qmbboot::operators {

inline constexpr int _PAULI_I = 0;
inline constexpr int _PAULI_X = 1;
inline constexpr int _PAULI_Z = 2;
inline constexpr int _PAULI_Y = 3;

template <std::size_t L_>
class PauliString {
    static_assert(L_ > 0, "L must be positive");

    public:
        static constexpr std::size_t L = L_;
        static constexpr std::size_t n_bits = 2 * L;
        static constexpr std::size_t n_words = (n_bits + 63) / 64;

    private:
        std::array<std::uint64_t, n_words> m_words{};

    public:
        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construction
        PauliString() = default;

        explicit PauliString(const std::array<std::uint64_t, n_words>& words) : m_words(words) {
            check_last_word();
        }

        explicit PauliString(const std::vector<std::uint64_t>& words) {
            if (words.size() > n_words) {
                for (std::size_t idx = n_words; idx < words.size(); ++idx) {
                    if (words[idx] != 0) {
                        throw std::invalid_argument("word vector exceeds 2L Pauli bits");
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

        static PauliString identity() {
            return PauliString();
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ comparison
        bool operator==(const PauliString& other) const {
            return m_words == other.m_words;
        }

        bool less(const PauliString& other) const {
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
        static PauliString from_string(const std::string& s) {
            if (s.size() != L) {
                throw std::invalid_argument("Pauli string length does not match L");
            }

            PauliString pstr;
            for (std::size_t site = 0; site < L; ++site) {
                pstr.set_code(site, pauli_code(s[site]));
            }
            return pstr;
        }

        std::string to_string() const {
            static constexpr char paulis[4] = {'I', 'X', 'Z', 'Y'};
            std::string out;
            out.reserve(L);
            for (std::size_t site = 0; site < L; ++site) {
                out.push_back(paulis[code(site)]);
            }
            return out;
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ physical properties
        int parity() const {
            int y_count = 0;
            for (std::size_t site = 0; site < L; ++site) {
                if (code(site) == _PAULI_Y) {
                    ++y_count;
                }
            }
            return y_count & 1;
        }

        std::array<int, 3> sign_charge() const {
            std::array<int, 3> charge{};
            for (std::size_t site = 0; site < L; ++site) {
                const auto c = code(site);
                if (c == _PAULI_X) {
                    charge[0] ^= 1;
                } else if (c == _PAULI_Y) {
                    charge[1] ^= 1;
                } else if (c == _PAULI_Z) {
                    charge[2] ^= 1;
                }
            }
            return charge;
        }

        std::pair<int, int> pi_rot_charge() const {
            const auto charge = sign_charge();
            return {(charge[0] + charge[1]) & 1, (charge[1] + charge[2]) & 1};
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ physical/symmetry operations
        PauliString translate(int shift) const {
            std::size_t shift_mod;
            if (shift >= 0) {
                shift_mod = static_cast<std::size_t>(shift) % L;
            } else {
                const auto abs_shift = static_cast<std::size_t>(-(shift + 1)) + 1;
                shift_mod = (L - abs_shift % L) % L;
            }
            if (shift_mod == 0) {
                return *this;
            }

            PauliString result;
            for (std::size_t site = 0; site < L; ++site) {
                result.set_code((site + shift_mod) % L, code(site));
            }
            return result;
        }

        PauliString trans_canon() const {
            auto canon = *this;
            for (std::size_t shift = 1; shift < L; ++shift) {
                auto cand = translate(static_cast<int>(shift));
                if (cand.less(canon)) {
                    canon = std::move(cand);
                }
            }
            return canon;
        }

        std::size_t period() const {
            for (std::size_t shift = 1; shift < L; ++shift) {
                if (translate(static_cast<int>(shift)) == *this) {
                    return shift;
                }
            }
            return L;
        }

        PauliString invert() const {
            PauliString result;
            for (std::size_t site = 0; site < L; ++site) {
                result.set_code((L - site) % L, code(site));
            }
            return result;
        }

        PauliString permute(const std::array<int, 3>& perm) const {
            PauliString result;
            for (std::size_t site = 0; site < L; ++site) {
                const auto c = code(site);
                if (c == _PAULI_X) {
                    result.set_code(site, perm[0]);
                } else if (c == _PAULI_Y) {
                    result.set_code(site, perm[1]);
                } else if (c == _PAULI_Z) {
                    result.set_code(site, perm[2]);
                }
            }
            return result;
        }

        std::pair<PauliString, std::complex<double>> mul(const PauliString& other) const {
            static constexpr int phase_power[4][4] = {
                {0, 0, 0, 0},
                {0, 0, 3, 1},
                {0, 1, 0, 3},
                {0, 3, 1, 0},
            };
            static constexpr std::complex<double> phase[4] = {
                {1.0, 0.0},
                {0.0, 1.0},
                {-1.0, 0.0},
                {0.0, -1.0},
            };

            PauliString result;
            int power = 0;
            for (std::size_t idx = 0; idx < n_words; ++idx) {
                result.m_words[idx] = m_words[idx] ^ other.m_words[idx];
            }
            for (std::size_t site = 0; site < L; ++site) {
                power += phase_power[code(site)][other.code(site)];
            }
            return {result, phase[power & 3]};
        }

        PauliString dag() const {
            return *this;
        }

    private:
        static int pauli_code(char pauli) {
            if (pauli == 'I') {
                return _PAULI_I;
            }
            if (pauli == 'X') {
                return _PAULI_X;
            }
            if (pauli == 'Z') {
                return _PAULI_Z;
            }
            if (pauli == 'Y') {
                return _PAULI_Y;
            }
            throw std::invalid_argument(std::string("invalid Pauli operator: ") + pauli);
        }

        void check_last_word() const {
            const auto rem = n_bits % 64;
            const auto last_mask = rem == 0 ? ~std::uint64_t{0} : ((std::uint64_t{1} << rem) - 1);
            if ((m_words.back() & ~last_mask) != 0) {
                throw std::invalid_argument("word vector exceeds 2L Pauli bits");
            }
        }

        void check_site(std::size_t site) const {
            if (site >= L) {
                throw std::out_of_range("Pauli site index out of range");
            }
        }

        int code(std::size_t site) const {
            check_site(site);
            const auto bit = 2 * site;
            return static_cast<int>((m_words[bit / 64] >> (bit % 64)) & 3);
        }

        void set_code(std::size_t site, int code) {
            check_site(site);
            if (code < 0 || code > 3) {
                throw std::invalid_argument("invalid Pauli code");
            }
            const auto bit = 2 * site;
            auto& word = m_words[bit / 64];
            word &= ~(std::uint64_t{3} << (bit % 64));
            word |= static_cast<std::uint64_t>(code) << (bit % 64);
        }
};

}  // namespace qmbboot::operators
