/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#pragma once

#include <rmm/detail/error.hpp>

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

/**
 * @brief Strongly-typed non-owning wrapper for CUDA streams with default constructor.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.
 */
class cuda_stream_view {
 public:
  constexpr cuda_stream_view()                        = default;
  constexpr cuda_stream_view(cuda_stream_view const&) = default;
  constexpr cuda_stream_view(cuda_stream_view&&)      = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view const&) = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view&&) = default;
  ~cuda_stream_view()                                       = default;

  // Disable construction from literal 0
  constexpr cuda_stream_view(int)            = delete;  //< Prevent cast from 0
  constexpr cuda_stream_view(std::nullptr_t) = delete;  //< Prevent cast from nullptr

  /**
   * @brief Implicit conversion from hipStream_t.
   */
  constexpr cuda_stream_view(hipStream_t stream) noexcept : stream_{stream} {}

  /**
   * @brief Get the wrapped stream.
   *
   * @return hipStream_t The wrapped stream.
   */
  [[nodiscard]] constexpr hipStream_t value() const noexcept { return stream_; }

  /**
   * @brief Implicit conversion to hipStream_t.
   */
  constexpr operator hipStream_t() const noexcept { return value(); }

  /**
   * @brief Return true if the wrapped stream is the CUDA per-thread default stream.
   */
  [[nodiscard]] inline bool is_per_thread_default() const noexcept;

  /**
   * @brief Return true if the wrapped stream is explicitly the CUDA legacy default stream.
   */
  [[nodiscard]] inline bool is_default() const noexcept;

  /**
   * @brief Synchronize the viewed CUDA stream.
   *
   * Calls `hipStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() const { RMM_CUDA_TRY(hipStreamSynchronize(stream_)); }

  /**
   * @brief Synchronize the viewed CUDA stream. Does not throw if there is an error.
   *
   * Calls `hipStreamSynchronize()` and asserts if there is an error.
   */
  void synchronize_no_throw() const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(hipStreamSynchronize(stream_));
  }

 private:
  hipStream_t stream_{};
};

/**
 * @brief Static cuda_stream_view of the default stream (stream 0), for convenience
 */
static constexpr cuda_stream_view cuda_stream_default{};

/**
 * @brief Static cuda_stream_view of cudaStreamLegacy, for convenience
 */

static const cuda_stream_view cuda_stream_legacy{
  cudaStreamLegacy  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
};

/**
 * @brief Static cuda_stream_view of hipStreamPerThread, for convenience
 */
static const cuda_stream_view cuda_stream_per_thread{
  hipStreamPerThread  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
};

[[nodiscard]] inline bool cuda_stream_view::is_per_thread_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return value() == cuda_stream_per_thread || value() == nullptr;
#else
  return value() == cuda_stream_per_thread;
#endif
}

/**
 * @brief Return true if the wrapped stream is explicitly the CUDA legacy default stream.
 */
[[nodiscard]] inline bool cuda_stream_view::is_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return value() == cuda_stream_legacy;
#else
  return value() == cuda_stream_legacy || value() == nullptr;
#endif
}

/**
 * @brief Equality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if equal, false if unequal
 */
inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs)
{
  return lhs.value() == rhs.value();
}

/**
 * @brief Inequality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if unequal, false if equal
 */
inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { return not(lhs == rhs); }

/**
 * @brief Output stream operator for printing / logging streams
 *
 * @param os The output ostream
 * @param sv The cuda_stream_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, cuda_stream_view stream)
{
  os << stream.value();
  return os;
}

}  // namespace rmm
