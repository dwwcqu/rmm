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

#include <cstddef>
#include <utility>

#include <cuda/memory_resource>

namespace rmm::mr {

/**
 * @brief Base class for host memory allocation.
 *
 * This is based on `std::pmr::memory_resource`:
 * https://en.cppreference.com/w/cpp/memory/memory_resource
 *
 * This class acts as a convenience utility class that handles equality_comparable_with and 
 * defines the `host_accessible` property
 *
 */
template <class Derived>
class host_memory_resource {
 public:
  host_memory_resource()                            = default;
  virtual ~host_memory_resource()                   = default;
  host_memory_resource(host_memory_resource const&) = default;
  host_memory_resource& operator=(host_memory_resource const&) = default;
  host_memory_resource(host_memory_resource&&) noexcept        = default;
  host_memory_resource& operator=(host_memory_resource&&) noexcept = default;

  /**
   * @brief Compare this resource to another.
   *
   * Two host_memory_resources compare equal if and only if memory allocated from one
   * host_memory_resource can be deallocated from the other and vice versa.
   *
   * By default, simply checks if `left` and `right` refer to the same object, i.e., does not check
   * whether they are two objects of the same class.
   *
   * @param left This resource
   * @param right The other resource to compare to
   * @return true If the two resources are equivalent
   */
  template<class Other, std::enable_if_t<std::is_same_v<Other, Derived>, int> = 0>
  [[nodiscard]] friend bool operator==(const Derived& left, const Other& right) noexcept {
      return &left == &right;
  }

  /**
   * @brief Compare this resource to another.
   * 
   * This synthesizes the inequality operator in case there is non defined but equality is defined
   *
   * @param left A resource of derived type
   * @param right A different compatible resource
   * @returns If the two resources are not equivalent
   */
  template<class T, class = std::void_t<decltype(std::declval<const Derived&>() == std::declval<const T&>())>>
  [[nodiscard]] friend bool operator!=(const Derived& left, const T& right) noexcept {
      return !(left == right);
  }
  
  /**
   * @brief Signal that this resource allocates host accessible memory.
   */
  friend void get_property(Derived const&, cuda::mr::host_accessible) noexcept
  {}
};
}  // namespace rmm::mr
