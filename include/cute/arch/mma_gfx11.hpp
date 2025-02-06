#pragma once

// hip passed

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

// Config
#if defined(__GFX11__) && __AMDGCN_WAVEFRONT_SIZE__ == 32
#  define CUTE_ARCH_MMA_GFX11_W32_ENABLED
#endif

namespace cute {

namespace {

using float_8 = float __attribute__((ext_vector_type(8)));

using half_t_16 = _Float16 __attribute__((ext_vector_type(16)));

using bfloat16_t_16 = short __attribute__((ext_vector_type(16)));

using int32_t_8 = int32_t __attribute__((ext_vector_type(8)));

using uint32_t_8 = uint32_t __attribute__((ext_vector_type(8)));

using int8_t_16 = int8_t __attribute__((ext_vector_type(16)));

using uint8_t_16 = uint8_t __attribute__((ext_vector_type(16)));

using int4_t_16 = int8_t __attribute__((ext_vector_type(8)));

using uint4_t_16 = uint8_t __attribute__((ext_vector_type(8)));

}

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TN
struct GFX11_16x16x16_F32F16F16F32_TN
{
    using DRegisters = float_8[1];
    using ARegisters = half_t_16[1];
    using BRegisters = half_t_16[1];
    using CRegisters = float_8[1];

    CUTE_HOST_DEVICE static void
    fma(float_8          & d0,
        half_t_16   const& a0,
        half_t_16   const& b0,
        float_8     const& c0)
    {
#if defined(CUTE_ARCH_MMA_GFX11_W32_ENABLED)
        d0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c0);
#else
        CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_F32F16F16F32_TN without CUTE_ARCH_MMA_GFX11_W32_ENABLED");
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TN
struct GFX11_16x16x16_F32BF16BF16F32_TN
{
    using DRegisters = float_8[1];
    using ARegisters = bfloat16_t_16[1];
    using BRegisters = bfloat16_t_16[1];
    using CRegisters = float_8[1];

    CUTE_HOST_DEVICE static void
    fma(float_8              & d0,
        bfloat16_t_16   const& a0,
        bfloat16_t_16   const& b0,
        float_8         const& c0)
    {
#if defined(CUTE_ARCH_MMA_GFX11_W32_ENABLED)
        d0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, c0);
#else
        CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_F32F16F16F32_TN without CUTE_ARCH_MMA_GFX11_W32_ENABLED");
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TN
template <bool opsel = false>
struct GFX11_16x16x16_F16F16F16F16_TN
{
  using DRegisters = half_t_16[1];
  using ARegisters = half_t_16[1];
  using BRegisters = half_t_16[1];
  using CRegisters = half_t_16[1];

  CUTE_HOST_DEVICE static void
  fma(half_t_16        & d0,
      half_t_16   const& a0,
      half_t_16   const& b0,
      half_t_16   const& c0)
  {
#if defined(CUTE_ARCH_MMA_GFX11_W32_ENABLED)
    d0 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a0, b0, c0, opsel);
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_F16F16F16F16_TN without CUTE_ARCH_MMA_GFX11_W32_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TIED TN
template <bool opsel = false>
struct GFX11_16x16x16_F16F16F16F16_TIED_TN
{
    using DRegisters = half_t_16[1];
    using ARegisters = half_t_16[1];
    using BRegisters = half_t_16[1];
    using CRegisters = half_t_16[1];

    CUTE_HOST_DEVICE static void
    fma(half_t_16        & d0,
        half_t_16   const& a0,
        half_t_16   const& b0,
        half_t_16   const& c0)
    {
#if defined(CUTE_ARCH_MMA_GFX11_W32_ENABLED)
        /*
        *  https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20231023/1228157.html
        *
        *  These new intrinsics, `amdgcn_wmma_tied_f16_16x16x16_f16` and `amdgcn_wmma_tied_f16_16x16x16_f16`,
        *  explicitly tie the destination accumulator matrix to the input accumulator matrix.
        *
        *  The `wmma_f16` and `wmma_bf16` intrinsics only write to 16-bit of the 32-bit destination VGPRs. 
        *  Which half is determined via the `op_sel` argument. The other half of the destination registers remains unchanged.
        *
        *  In some cases however, we expect the destination to copy the other halves from the input accumulator.
        *  For instance, when packing two separate accumulator matrices into one. In that case, the two matrices 
        *  are tied into the same registers, but separate halves. Then it is important to copy the other matrix values
        *  to the new destination.
        */
        d0 = __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(a0, b0, c0, opsel);
#else
        CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_F16F16F16F16_TIED_TN without CUTE_ARCH_MMA_GFX11_W32_ENABLED");
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
template <bool opsel = false>
struct GFX11_16x16x16_BF16BF16BF16BF16_TN
{
    using DRegisters = bfloat16_t_16[1];
    using ARegisters = bfloat16_t_16[1];
    using BRegisters = bfloat16_t_16[1];
    using CRegisters = bfloat16_t_16[1];

    CUTE_HOST_DEVICE static void
    fma(bfloat16_t_16        & d0,
        bfloat16_t_16   const& a0,
        bfloat16_t_16   const& b0,
        bfloat16_t_16   const& c0)
    {
#if defined(CUTE_ARCH_MMA_GFX11_W32_ENABLED)
        d0 = __builtin_amdgcn_wmma_bfloat16_t_16x16x16_bf16_w32(a0, b0, c0, opsel);
#else
        CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_BF16BF16BF16BF16_TN without CUTE_ARCH_MMA_GFX11_W32_ENABLED");
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TIED TN
template <bool opsel = false>
struct GFX11_16x16x16_BF16BF16BF16BF16_TIED_TN
{
    using DRegisters = bfloat16_t_16[1];
    using ARegisters = bfloat16_t_16[1];
    using BRegisters = bfloat16_t_16[1];
    using CRegisters = bfloat16_t_16[1];

    CUTE_HOST_DEVICE static void
    fma(bfloat16_t_16        & d0,
        bfloat16_t_16   const& a0,
        bfloat16_t_16   const& b0,
        bfloat16_t_16   const& c0)
    {
#if defined(CUTE_ARCH_MMA_GFX11_W32_ENABLED)
        /*
        *  check out the comments in GFX11_16x16x16_F16F16F16F16_TIED_TN
        */
        d0 = __builtin_amdgcn_wmma_bfloat16_t_16x16x16_bf16_tied_w32(a0, b0, c0, opsel);
#else
        CUTE_INVALID_CONTROL_PATH("Attempting to use GFX11_16x16x16_BF16BF16BF16BF16_TIED_TN without CUTE_ARCH_MMA_GFX11_W32_ENABLED");
#endif
    }
};

}