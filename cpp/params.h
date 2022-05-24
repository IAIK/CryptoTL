#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <stdint.h>

namespace HE_TL_PARAM {
// CKKS parameters
#if defined(ACCURATE_PARAMS)
constexpr uint64_t MOD_DEGREE = 16384;
constexpr uint64_t BIT_SCALE = 50;       // scale for CKKS
constexpr uint64_t CKKS_FIX_SCALE = 60;  // scale for CKKS special primes
#else
constexpr uint64_t MOD_DEGREE = 8192;
constexpr uint64_t BIT_SCALE = 25;       // scale for CKKS
constexpr uint64_t CKKS_FIX_SCALE = 34;  // scale for CKKS special primes
#endif
// Babystep giantstep multiplication
constexpr bool USE_BSGS = true;
}  // namespace HE_TL_PARAM

#endif  // _PARAMS_H_
