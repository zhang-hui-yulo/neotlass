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

#if !defined(__HIPCC_RTC__)
#include "hip/hip_runtime.h"
#include "cutlass/trace.h"
#endif
#include <cute/int_tuple.hpp>

namespace cutlass {

struct KernelHardwareInfo {
  //
  // Data members
  //

  // Hardware properties
  int device_id = 0;
  int sm_count  = 0;

  //
  // Methods
  //

#if !defined(__HIPCC_RTC__)
  static inline int
  query_device_multiprocessor_count(int device_id = 0) {
    hipError_t result = hipGetDevice(&device_id);
    if (result != hipSuccess) {
      CUTLASS_TRACE_HOST(
        "  hipGetDevice() returned error "
        << hipGetErrorString(result));
      return 0;
    }
    int multiprocessor_count;
    result = hipDeviceGetAttribute(&multiprocessor_count,
      hipDeviceAttributeMultiprocessorCount, device_id);
    if (result != hipSuccess) {
      CUTLASS_TRACE_HOST(
        "  hipDeviceGetAttribute() returned error "
        << hipGetErrorString(result));
      return 0;
    }
    return multiprocessor_count;
  }
#endif
};

} // namespace cutlass
