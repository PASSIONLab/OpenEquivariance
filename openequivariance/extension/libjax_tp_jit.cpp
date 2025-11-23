/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/*
ffi::Error ArrayAttrImpl(ffi::Span<const int32_t> array,
                         ffi::ResultBufferR0<ffi::S32> res) {
  int64_t total = 0;
  for (int32_t x : array) {
    total += x;
  }
  res->typed_data()[0] = total;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ArrayAttr, ArrayAttrImpl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Span<const int32_t>>("array")
                                  .Ret<ffi::BufferR0<ffi::S32>>());

ffi::Error DictionaryAttrImpl(ffi::Dictionary attrs,
                              ffi::ResultBufferR0<ffi::S32> secret,
                              ffi::ResultBufferR0<ffi::S32> count) {
  auto maybe_secret = attrs.get<int64_t>("secret");
  if (maybe_secret.has_error()) {
    return maybe_secret.error();
  }
  secret->typed_data()[0] = maybe_secret.value();
  count->typed_data()[0] = attrs.size();
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DictionaryAttr, DictionaryAttrImpl,
                              ffi::Ffi::Bind()
                                  .Attrs()
                                  .Ret<ffi::BufferR0<ffi::S32>>()
                                  .Ret<ffi::BufferR0<ffi::S32>>());
*/

ffi::Error tp_forward_impl(
  cudaStream_t stream, 
  std::string_view kernel, 
  ffi::Dictionary forward_config, 
  ffi::ResultBufferR0<ffi::S32> out) {
  static std::mutex mutex;
  static auto &cache = *new std::unordered_map<std::string_view, int32_t>();

  auto value = forward_config.get<int64_t>("example_key").value();
  std::cout << value << std::endl;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    /*auto it = cache.find(key);
    if (it != cache.end()) {
      out->typed_data()[0] = ++it->second;
    } else {
      cache.insert({key, 0});
      out->typed_data()[0] = 0;
    }*/
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tp_forward, tp_forward_impl,
    ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<std::string_view>("kernel")
      .Attr<ffi::Dictionary>("forward_config")
      .Ret<ffi::BufferR0<ffi::S32>>(),
      {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

NB_MODULE(oeq_jax_extension, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["tp_forward"] = nb::capsule(reinterpret_cast<void *>(tp_forward));
    return registrations;
  });
}
