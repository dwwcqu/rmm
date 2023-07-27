/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "synchronization.hpp"

#include <rmm/device_buffer.hpp>

#ifdef NDEBUG
#define RMM_CUDA_ASSERT_OK(expr) expr
#else
#define RMM_CUDA_ASSERT_OK(expr)       \
  do {                                 \
    hipError_t const status = (expr); \
    assert(hipSuccess == status);     \
  } while (0);
#endif

cuda_event_timer::cuda_event_timer(benchmark::State& state,
                                   bool flush_l2_cache,
                                   rmm::cuda_stream_view stream)
  : stream(stream), p_state(&state)
{
  // flush all of L2$
  if (flush_l2_cache) {
    int current_device = 0;
    RMM_CUDA_TRY(hipGetDevice(&current_device));

    int l2_cache_bytes = 0;
    RMM_CUDA_TRY(hipDeviceGetAttribute(&l2_cache_bytes, hipDeviceAttributeL2CacheSize, current_device));

    if (l2_cache_bytes > 0) {
      const int memset_value = 0;
      rmm::device_buffer l2_cache_buffer(l2_cache_bytes, stream);
      RMM_CUDA_TRY(
        hipMemsetAsync(l2_cache_buffer.data(), memset_value, l2_cache_bytes, stream.value()));
    }
  }

  RMM_CUDA_TRY(hipEventCreate(&start));
  RMM_CUDA_TRY(hipEventCreate(&stop));
  RMM_CUDA_TRY(hipEventRecord(start, stream.value()));
}

cuda_event_timer::~cuda_event_timer()
{
  RMM_CUDA_ASSERT_OK(hipEventRecord(stop, stream.value()));
  RMM_CUDA_ASSERT_OK(hipEventSynchronize(stop));

  float milliseconds = 0.0F;
  RMM_CUDA_ASSERT_OK(hipEventElapsedTime(&milliseconds, start, stop));
  const auto to_milliseconds{1.0F / 1000};
  p_state->SetIterationTime(milliseconds * to_milliseconds);
  RMM_CUDA_ASSERT_OK(hipEventDestroy(start));
  RMM_CUDA_ASSERT_OK(hipEventDestroy(stop));
}
