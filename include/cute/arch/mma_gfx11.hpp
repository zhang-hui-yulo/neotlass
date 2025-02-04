#pragma once

// hip passed

#include <cute/config.hpp>

// Config
#if defined(__GFX11__)
#  define CUTE_ARCH_MMA_GFX11_ENABLED
#endif

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TN
template <bool opsel = false>
struct GFX11_16x16x16_F16F16F16F16_TN
{
  using half_t16 = _Float16 __attribute__((ext_vector_type(16)));
  using DRegisters = half_t16[1];
  using ARegisters = half_t16[1];
  using BRegisters = half_t16[1];
  using CRegisters = half_t16[1];

  CUTE_HOST_DEVICE static void
  fma(half_t16        & d0,
      half_t16   const& a0,
      half_t16   const& b0,
      half_t16   const& c0)
  {
#if defined(CUTE_ARCH_MMA_GFX11_ENABLED)
    d0 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a0, b0, c0, opsel);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_F16F16F16F16_TN without CUTE_ARCH_MMA_GFX10_ENABLED");
#endif
  }
};

template <bool opsel = false>
struct GFX11_16x16x16_BF16BF16BF16BF16_TN
{
    typedef unsigned short bf16_16 __attribute__((ext_vector_type(16)));
    using DRegisters = bf16_16[1];
    using ARegisters = bf16_16[1];
    using BRegisters = bf16_16[1];
    using CRegisters = bf16_16[1];

    CUTE_HOST_DEVICE static void
        fma(bf16_16        & d0,
            bf16_16   const& a0,
            bf16_16   const& b0,
            bf16_16   const& c0)
    {
#if defined(CUTE_ARCH_MMA_GFX11_ENABLED)
        d0 = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(a0, b0, c0, opsel);
#else
        CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_BF16BF16BF16BF16_TN without CUTE_ARCH_MMA_GFX10_ENABLED");
#endif
    }
};

}