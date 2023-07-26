/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <hip/hip_runtime_api.h>

#include <memory>

namespace rmm {

/**
 * @brief Owning wrapper for a CUDA stream.
 *
 * Provides RAII lifetime semantics for a CUDA stream.
 */
class hip_stream {
 public:
  /**
   * @brief Move constructor (default)
   *
   * A moved-from hip_stream is invalid and it is Undefined Behavior to call methods that access
   * the owned stream.
   */
  hip_stream(hip_stream&&) = default;
  /**
   * @brief Move copy assignment operator (default)
   *
   * A moved-from hip_stream is invalid and it is Undefined Behavior to call methods that access
   * the owned stream.
   */
  hip_stream& operator=(hip_stream&&) = default;
  ~hip_stream()                        = default;
  hip_stream(hip_stream const&)       = delete;  // Copying disallowed: one stream one owner
  hip_stream& operator=(hip_stream&) = delete;

  /**
   * @brief Construct a new cuda stream object
   *
   * @throw rmm::cuda_error if stream creation fails
   */
  hip_stream()
    : stream_{[]() {
                auto* stream = new hipStream_t;  // NOLINT(cppcoreguidelines-owning-memory)
                RMM_CUDA_TRY(hipStreamCreate(stream));
                return stream;
              }(),
              [](hipStream_t* stream) {
                RMM_ASSERT_CUDA_SUCCESS(hipStreamDestroy(*stream));
                delete stream;  // NOLINT(cppcoreguidelines-owning-memory)
              }}
  {
  }

  /**
   * @brief Returns true if the owned stream is non-null
   *
   * @return true If the owned stream has not been explicitly moved and is therefore non-null.
   * @return false If the owned stream has been explicitly moved and is therefore null.
   */
  [[nodiscard]] bool is_valid() const { return stream_ != nullptr; }

  /**
   * @brief Get the value of the wrapped CUDA stream.
   *
   * @return hipStream_t The wrapped CUDA stream.
   */
  [[nodiscard]] hipStream_t value() const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return *stream_;
  }

  /**
   * @brief Explicit conversion to hipStream_t.
   */
  explicit operator hipStream_t() const noexcept { return value(); }

  /**
   * @brief Creates an immutable, non-owning view of the wrapped CUDA stream.
   *
   * @return rmm::cuda_stream_view The view of the CUDA stream
   */
  [[nodiscard]] cuda_stream_view view() const { return cuda_stream_view{value()}; }

  /**
   * @brief Implicit conversion to cuda_stream_view
   *
   * @return A view of the owned stream
   */
  operator cuda_stream_view() const { return view(); }

  /**
   * @brief Synchronize the owned CUDA stream.
   *
   * Calls `hipStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() const { RMM_CUDA_TRY(hipStreamSynchronize(value())); }

  /**
   * @brief Synchronize the owned CUDA stream. Does not throw if there is an error.
   *
   * Calls `hipStreamSynchronize()` and asserts if there is an error.
   */
  void synchronize_no_throw() const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(hipStreamSynchronize(value()));
  }

 private:
  std::unique_ptr<hipStream_t, std::function<void(hipStream_t*)>> stream_;
};

}  // namespace rmm
