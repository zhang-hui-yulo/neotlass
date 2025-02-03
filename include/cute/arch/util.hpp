/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

// hip passed

#include <cute/config.hpp>
#include <cute/numeric/integer_sequence.hpp>


namespace cute
{

/// CUTE helper to cast SMEM pointer to unsigned
CUTE_DEVICE
uint32_t
cast_smem_ptr_to_uint(void const* const ptr)
{
  (void) ptr;
  printf("ERROR: cast_smem_ptr_to_uint not supported but used.\n");
  return 0;
}

namespace detail {

//
// Wrapper for MMAOp::fma
//

template <class MmaOp>
struct CallFMA {
  template <class... Args>
  CUTE_HOST_DEVICE constexpr void
  operator()(Args&&... args) const {
    return MmaOp::fma(static_cast<Args&&>(args)...);
  }
};

//
// Wrapper for CopyOp::copy
//

template <class CopyOp>
struct CallCOPY {
  template <class... Args>
  CUTE_HOST_DEVICE constexpr void
  operator()(Args&&... args) const {
    return CopyOp::copy(static_cast<Args&&>(args)...);
  }
};

//
// Utility for exploding pointers/arrays/tensors into functions
//

template <class Fn,
          class PtrA, int... I>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrA&& a, int_sequence<I...>)
{
  return fn(a[I]...);
}

template <class Fn,
          class PtrS, int... Is,
          class PtrD, int... Id>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrS&& s, int_sequence<Is...>,
        PtrD&& d, int_sequence<Id...>)
{
  return fn(s[Is]..., d[Id]...);
}

template <class Fn,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>)
{
  return fn(a[Ia]..., b[Ib]..., c[Ic]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic,
          class PtrE, int... Ie>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>,
        PtrE&& e, int_sequence<Ie...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]..., e[Ie]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic,
          class PtrE, int... Ie,
          class PtrF, int... If>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>,
        PtrE&& e, int_sequence<Ie...>,
        PtrF&& f, int_sequence<If...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]..., e[Ie]..., f[If]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic,
          class PtrE, int... Ie,
          class PtrF, int... If,
          class PtrG, int... Ig>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>,
        PtrE&& e, int_sequence<Ie...>,
        PtrF&& f, int_sequence<If...>,
        PtrG&& g, int_sequence<Ig...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]..., e[Ie]..., f[If]..., g[Ig]...);
}

//
// Utility for exploding tuples into functions
//

template <class Fn,
          class TupleA, int... I>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<I...>)
{
  return fn(get<I>(a)...);
}

template <class Fn,
          class TupleA, int... Ia,
          class TupleB, int... Ib>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<Ia...>,
              TupleB&& b, int_sequence<Ib...>)
{
  return fn(get<Ia>(a)..., get<Ib>(b)...);
}

template <class Fn,
          class TupleA, int... Ia,
          class TupleB, int... Ib,
          class TupleC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<Ia...>,
              TupleB&& b, int_sequence<Ib...>,
              TupleC&& c, int_sequence<Ic...>)
{
  return fn(get<Ia>(a)..., get<Ib>(b)..., get<Ic>(c)...);
}

} // end namespace detail

} // end namespace cute
