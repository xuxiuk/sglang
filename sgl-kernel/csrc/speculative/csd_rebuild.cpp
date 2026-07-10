/* Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr int64_t kCsdEmptyKey = -1;

uint64_t hash64(uint64_t key) {
  uint64_t x = key;
  x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
  return x ^ (x >> 31);
}

int64_t next_power_of_two(int64_t value) {
  int64_t result = 1;
  while (result < value) {
    TORCH_CHECK(
        result <= std::numeric_limits<int64_t>::max() / 2,
        "CSD hash table capacity overflow");
    result *= 2;
  }
  return result;
}

std::pair<bool, int64_t> try_build(
    const int64_t* keys,
    int64_t num_keys,
    int64_t capacity,
    int64_t max_probe,
    std::vector<int64_t>& table) {
  table.assign(capacity, kCsdEmptyKey);
  const uint64_t mask = static_cast<uint64_t>(capacity - 1);
  int64_t num_entries = 0;
  for (int64_t index = 0; index < num_keys; ++index) {
    const int64_t key = keys[index];
    uint64_t slot = hash64(static_cast<uint64_t>(key)) & mask;
    bool inserted = false;
    for (int64_t probe = 0; probe < max_probe; ++probe) {
      const int64_t existing = table[slot];
      if (existing == kCsdEmptyKey || existing == key) {
        num_entries += existing == kCsdEmptyKey;
        table[slot] = key;
        inserted = true;
        break;
      }
      slot = (slot + 1) & mask;
    }
    if (!inserted) {
      return {false, 0};
    }
  }
  return {true, num_entries};
}

}  // namespace

std::tuple<torch::Tensor, int64_t> csd_build_hash_table_cpu(
    const torch::Tensor& input_keys,
    int64_t max_probe,
    double load_factor);

class CSDTableBuilder : public torch::CustomClassHolder {
 public:
  CSDTableBuilder(
      const torch::Tensor& input_keys,
      const torch::Tensor& input_freqs,
      int64_t freq_threshold)
      : freq_threshold_(freq_threshold) {
    TORCH_CHECK(freq_threshold >= 1, "CSD frequency threshold must be positive");
    const torch::Tensor keys = checked_input(input_keys, "keys");
    const torch::Tensor freqs = checked_input(input_freqs, "frequencies");
    TORCH_CHECK(keys.numel() == freqs.numel(), "CSD keys/frequencies size mismatch");
    const int64_t* key_data = keys.const_data_ptr<int64_t>();
    const int64_t* freq_data = freqs.const_data_ptr<int64_t>();
    counts_.reserve(keys.numel());
    active_keys_.reserve(keys.numel() / 16);
    for (int64_t index = 0; index < keys.numel(); ++index) {
      TORCH_CHECK(key_data[index] >= 0, "CSD keys must be non-negative");
      TORCH_CHECK(freq_data[index] >= 0, "CSD frequencies must be non-negative");
      counts_[key_data[index]] = freq_data[index];
      if (freq_data[index] >= freq_threshold_) {
        active_keys_.insert(key_data[index]);
      }
    }
  }

  int64_t update(const torch::Tensor& input_pairs) {
    const torch::Tensor pairs = checked_input(input_pairs, "pairs");
    const int64_t* pair_data = pairs.const_data_ptr<int64_t>();
    for (int64_t index = 0; index < pairs.numel(); ++index) {
      const int64_t key = pair_data[index];
      if (key < 0) {
        continue;
      }
      int64_t& count = counts_[key];
      ++count;
      if (count == freq_threshold_) {
        active_keys_.insert(key);
      }
    }
    return counts_.size();
  }

  std::tuple<torch::Tensor, int64_t, int64_t> update_and_build(
      const torch::Tensor& pairs,
      int64_t max_probe,
      double load_factor) {
    update(pairs);
    std::vector<int64_t> keys(active_keys_.begin(), active_keys_.end());
    torch::Tensor key_tensor = torch::empty(
        {static_cast<int64_t>(keys.size())}, torch::TensorOptions().dtype(torch::kInt64));
    std::copy(keys.begin(), keys.end(), key_tensor.mutable_data_ptr<int64_t>());
    auto [table, num_entries] =
        csd_build_hash_table_cpu(key_tensor, max_probe, load_factor);
    return {table, num_entries, static_cast<int64_t>(counts_.size())};
  }

  std::tuple<torch::Tensor, torch::Tensor> snapshot() const {
    torch::Tensor keys = torch::empty(
        {static_cast<int64_t>(counts_.size())}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor freqs = torch::empty_like(keys);
    int64_t* key_data = keys.mutable_data_ptr<int64_t>();
    int64_t* freq_data = freqs.mutable_data_ptr<int64_t>();
    int64_t index = 0;
    for (const auto& [key, freq] : counts_) {
      key_data[index] = key;
      freq_data[index] = freq;
      ++index;
    }
    return {keys, freqs};
  }

  int64_t num_store_entries() const {
    return counts_.size();
  }

 private:
  static torch::Tensor checked_input(
      const torch::Tensor& input, const char* name) {
    TORCH_CHECK(input.device().is_cpu(), "CSD ", name, " must be on CPU");
    TORCH_CHECK(
        input.scalar_type() == torch::kInt64,
        "CSD ",
        name,
        " must be int64");
    TORCH_CHECK(input.dim() == 1, "CSD ", name, " must be one-dimensional");
    return input.contiguous();
  }

  int64_t freq_threshold_;
  std::unordered_map<int64_t, int64_t> counts_;
  std::unordered_set<int64_t> active_keys_;
};

std::tuple<torch::Tensor, int64_t> csd_build_hash_table_cpu(
    const torch::Tensor& input_keys,
    int64_t max_probe,
    double load_factor) {
  TORCH_CHECK(input_keys.device().is_cpu(), "CSD keys must be on CPU");
  TORCH_CHECK(
      input_keys.scalar_type() == torch::kInt64, "CSD keys must be int64");
  TORCH_CHECK(input_keys.dim() == 1, "CSD keys must be one-dimensional");
  TORCH_CHECK(max_probe >= 1, "CSD max_probe must be at least 1");
  TORCH_CHECK(
      load_factor > 0.0 && load_factor <= 1.0,
      "CSD load_factor must be in (0, 1]");

  const torch::Tensor keys = input_keys.contiguous();
  const int64_t num_keys = keys.numel();
  if (num_keys == 0) {
    return {torch::full({1}, kCsdEmptyKey, keys.options()), 0};
  }

  const int64_t* key_data = keys.const_data_ptr<int64_t>();
  for (int64_t index = 0; index < num_keys; ++index) {
    TORCH_CHECK(key_data[index] >= 0, "CSD keys must be non-negative");
  }

  const auto minimum_capacity = static_cast<int64_t>(
      std::ceil(static_cast<double>(num_keys) / load_factor));
  int64_t capacity = next_power_of_two(std::max<int64_t>(2, minimum_capacity));
  std::vector<int64_t> table;
  int64_t num_entries = 0;
  while (true) {
    auto [built, entries] =
        try_build(key_data, num_keys, capacity, max_probe, table);
    if (built) {
      num_entries = entries;
      break;
    }
    TORCH_CHECK(
        capacity <= std::numeric_limits<int64_t>::max() / 2,
        "CSD hash table capacity overflow");
    capacity *= 2;
  }

  torch::Tensor output = torch::empty({capacity}, keys.options());
  std::copy(table.begin(), table.end(), output.mutable_data_ptr<int64_t>());
  return {output, num_entries};
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "csd_build_hash_table_cpu(Tensor keys, int max_probe, float load_factor) "
      "-> (Tensor, int)");
  m.impl("csd_build_hash_table_cpu", torch::kCPU, &csd_build_hash_table_cpu);
  m.class_<CSDTableBuilder>("CSDTableBuilder")
      .def(torch::init<torch::Tensor, torch::Tensor, int64_t>())
      .def("update", &CSDTableBuilder::update)
      .def("update_and_build", &CSDTableBuilder::update_and_build)
      .def("snapshot", &CSDTableBuilder::snapshot)
      .def("num_store_entries", &CSDTableBuilder::num_store_entries);
}
