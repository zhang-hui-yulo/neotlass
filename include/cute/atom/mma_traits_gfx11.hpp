#pragma once

// hip passed

#include <cute/arch/mma_gfx11.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

namespace {

// (T32,V16) -> (M16,N16)
using GFX11_16x16_Row = Layout<Shape <Shape <_16,_2>,Shape < _16>>,
                               Stride<Stride< _1,_0>,Stride< _16>>>;

template <bool opsel>
struct CUTE_ALIGNAS(4) GFX11FrgTypeAccum {
    half_t data[2];

    CUTE_HOST_DEVICE constexpr operator half_t() const noexcept {
        if constexpr (!opsel) {
            return data[0];
        } else {
            return data[1];
        }
    }
};

}

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<false>>
{
    using ValTypeD = half_t;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = half_t;

    using FrgTypeC = GFX11FrgTypeAccum<false>;

    using Shape_MNK = Shape<_16, _16, _16>;
    using ThrID = Layout<_32>;
    using ALayout = GFX11_16x16_Row;
    using BLayout = GFX11_16x16_Row;
    using CLayout = Layout<Shape <Shape <_16, _2>, Shape < _8>>,
                           Stride<Stride<_16, _1>, Stride< _2>>>;
};

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<true>>
     : MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<false>> {
    using FrgTypeC = GFX11FrgTypeAccum<true>;
};

}