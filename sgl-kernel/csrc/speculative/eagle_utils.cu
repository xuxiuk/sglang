/*
 * Copyright (c) 2025 by SGLang team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] =
// [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token] positions [bs * draft_token] retrive_index [b,
// draft_token] retrive_next_token [b, draft_token] retrive_next_sibling [b, draft_token]
__global__ void build_tree_efficient(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
  }
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask: [draft_token*num_bytes_per_item | .. ] = [bs*draft_token*num_bytes_per_item]
// positions [bs * draft_token]
// retrive_index [bs, draft_token]
// retrive_next_token [bs, draft_token]
// retrive_next_sibling [bs, draft_token]
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    uint8_t* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    size_t num_bytes_per_item) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item;
  tree_mask[token_tree_idx] = 1;  // little endian

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      int byte_idx = (cur_position + 1) / 8;
      int bit_idx = (cur_position + 1) % 8;
      tree_mask[token_tree_idx + byte_idx] |= (1 << bit_idx);
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

void build_tree_kernel_efficient(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  // TODO (ying) check shape
  // TODO (ying) check type
  int bs = parent_list.size(0);
  dim3 grid(bs);
  dim3 block(draft_token_num);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (tree_mask_mode == QLEN_ONLY_BITPACKING) {
    size_t num_bytes_per_item = 1;
    if (draft_token_num > 16) {
      num_bytes_per_item = 4;
    } else if (draft_token_num > 8) {
      num_bytes_per_item = 2;
    }
    build_tree_efficient_partial_packed<<<grid, block, 0, stream>>>(
        static_cast<int64_t*>(parent_list.data_ptr()),
        static_cast<int64_t*>(selected_index.data_ptr()),
        static_cast<int64_t*>(verified_seq_len.data_ptr()),
        static_cast<uint8_t*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrive_index.data_ptr()),
        static_cast<int64_t*>(retrive_next_token.data_ptr()),
        static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
        int32_t(topk),
        int32_t(depth),
        int32_t(draft_token_num),
        num_bytes_per_item);
  } else {
    build_tree_efficient<<<grid, block, 0, stream>>>(
        static_cast<int64_t*>(parent_list.data_ptr()),
        static_cast<int64_t*>(selected_index.data_ptr()),
        static_cast<int64_t*>(verified_seq_len.data_ptr()),
        static_cast<bool*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrive_index.data_ptr()),
        static_cast<int64_t*>(retrive_next_token.data_ptr()),
        static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
        int32_t(topk),
        int32_t(depth),
        int32_t(draft_token_num),
        int32_t(tree_mask_mode));
  }
}

static constexpr int64_t CSD_EMPTY_KEY = -1;

__device__ __forceinline__ uint64_t CsdHash64(uint64_t key) {
  key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
  key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
  return key ^ (key >> 31);
}

__device__ __forceinline__ int64_t CsdPackPair(int64_t lhs_token, int64_t rhs_token) {
  return static_cast<int64_t>(
      (static_cast<uint64_t>(lhs_token) << 32) | (static_cast<uint64_t>(rhs_token) & 0xffffffffULL));
}

__device__ __forceinline__ bool CsdHashContains(
    const int64_t* keys,
    uint32_t capacity,
    uint32_t max_probe,
    int64_t key) {
  if (capacity == 0 || key == CSD_EMPTY_KEY) {
    return false;
  }

  uint32_t slot = static_cast<uint32_t>(
      CsdHash64(static_cast<uint64_t>(key)) & static_cast<uint64_t>(capacity - 1));
  for (uint32_t probe = 0; probe < max_probe; ++probe) {
    int64_t existing_key = keys[slot];
    if (existing_key == key) {
      return true;
    }
    if (existing_key == CSD_EMPTY_KEY) {
      return false;
    }
    slot = (slot + 1) & (capacity - 1);
  }
  return false;
}

__device__ __forceinline__ void CsdAtomicAddI64(int64_t* value, unsigned long long increment) {
  atomicAdd(reinterpret_cast<unsigned long long*>(value), increment);
}

__device__ __forceinline__ void CsdAppendDelta(
    int64_t* delta_pairs,
    int32_t* delta_counter,
    int64_t* delta_pair_ct,
    uint32_t delta_capacity,
    int64_t key) {
  if (delta_capacity == 0 || key == CSD_EMPTY_KEY) {
    return;
  }

  int32_t pos = atomicAdd(delta_counter, 1);
  if (pos >= 0 && static_cast<uint32_t>(pos) < delta_capacity) {
    delta_pairs[pos] = key;
    CsdAtomicAddI64(delta_pair_ct, 1ULL);
  }
}

template <typename DType, typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    DType* target_logits,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    const int64_t* csd_table_keys,
    int64_t* csd_delta_pairs,
    int32_t* csd_delta_counter,
    int64_t* csd_lookup_hit_ct,
    int64_t* csd_forced_accept_ct,
    int64_t* csd_delta_pair_ct,
    uint32_t csd_table_capacity,
    uint32_t csd_table_max_probe,
    uint32_t csd_delta_capacity,
    bool csd_enabled,
    bool csd_dynamic_update,
    bool csd_dynamic_update_ignore_prob_ratio,
    bool csd_force_accept_disabled,
    DType csd_logit_margin) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      bool normal_accept = draft_token_id == target_token_id;
      bool csd_force_accept = false;

      if (!normal_accept && (csd_enabled || csd_dynamic_update)) {
        int64_t csd_pair_key = CsdPackPair(draft_token_id, target_token_id);
        size_t target_logit_offset = static_cast<size_t>(last_accepted_retrive_idx) * d;
        DType draft_logit = target_logits[target_logit_offset + draft_token_id];
        DType target_logit = target_logits[target_logit_offset + target_token_id];
        bool csd_logit_pass = draft_logit >= target_logit + csd_logit_margin;
        if (csd_dynamic_update && (csd_dynamic_update_ignore_prob_ratio || csd_logit_pass)) {
          CsdAppendDelta(csd_delta_pairs, csd_delta_counter, csd_delta_pair_ct, csd_delta_capacity, csd_pair_key);
        }

        bool table_hit =
            csd_enabled && CsdHashContains(csd_table_keys, csd_table_capacity, csd_table_max_probe, csd_pair_key);
        if (table_hit) {
          CsdAtomicAddI64(csd_lookup_hit_ct, 1ULL);
        }
        if (table_hit && !csd_force_accept_disabled) {
          csd_force_accept = csd_logit_pass;
          if (csd_force_accept) {
            CsdAtomicAddI64(csd_forced_accept_ct, 1ULL);
          }
        }
      }

      if (normal_accept || csd_force_accept) {
        // accept token
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, num_spec_step]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// target_predict: [bs, num_draft_tokens]
// target_logits: [bs, num_draft_tokens, vocab_size]
void verify_tree_greedy(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,  // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict,
    at::Tensor target_logits,
    at::Tensor csd_table_keys,
    at::Tensor csd_delta_pairs,
    at::Tensor csd_delta_counter,
    at::Tensor csd_lookup_hit_ct,
    at::Tensor csd_forced_accept_ct,
    at::Tensor csd_delta_pair_ct,
    int64_t csd_table_capacity,
    int64_t csd_table_max_probe,
    int64_t csd_delta_capacity,
    bool csd_enabled,
    bool csd_dynamic_update,
    bool csd_dynamic_update_ignore_prob_ratio,
    bool csd_force_accept_disabled,
    double csd_logit_margin) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(target_predict);
  CHECK_INPUT(target_logits);
  CHECK_INPUT(csd_table_keys);
  CHECK_INPUT(csd_delta_pairs);
  CHECK_INPUT(csd_delta_counter);
  CHECK_INPUT(csd_lookup_hit_ct);
  CHECK_INPUT(csd_forced_accept_ct);
  CHECK_INPUT(csd_delta_pair_ct);
  auto device = target_predict.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(target_predict.device(), device);
  CHECK_EQ(target_logits.device(), device);
  CHECK_EQ(csd_table_keys.device(), device);
  CHECK_EQ(csd_delta_pairs.device(), device);
  CHECK_EQ(csd_delta_counter.device(), device);
  CHECK_EQ(csd_lookup_hit_ct.device(), device);
  CHECK_EQ(csd_forced_accept_ct.device(), device);
  CHECK_EQ(csd_delta_pair_ct.device(), device);
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, target_predict);
  CHECK_DIM(3, target_logits);
  CHECK_DIM(1, csd_table_keys);
  CHECK_DIM(1, csd_delta_pairs);
  CHECK_DIM(1, csd_delta_counter);
  CHECK_DIM(1, csd_lookup_hit_ct);
  CHECK_DIM(1, csd_forced_accept_ct);
  CHECK_DIM(1, csd_delta_pair_ct);
  unsigned int batch_size = candidates.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_logits.size(2);
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(batch_size, target_predict.size(0));
  CHECK_EQ(batch_size, target_logits.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(num_draft_tokens, target_predict.size(1));
  CHECK_EQ(num_draft_tokens, target_logits.size(1));
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));
  if (predicts.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'predicts' to be of type int (torch.int32).");
  }
  if (accept_index.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_index' to be of type int (torch.int32).");
  }
  if (accept_token_num.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_token_num' to be of type int (torch.int32).");
  }
  if (candidates.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'candidates' to be of type long (torch.int64).");
  }
  if (retrive_index.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'retrive_index' to be of type long (torch.int64).");
  }
  if (retrive_next_token.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'retrive_next_token' to be of type long (torch.int64).");
  }
  if (retrive_next_sibling.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'retrive_next_sibling' to be of type long (torch.int64).");
  }
  if (target_predict.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'target_predict' to be of type long (torch.int64).");
  }
  if (target_logits.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'target_logits' to be of type float (torch.float32).");
  }
  if (csd_table_keys.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'csd_table_keys' to be of type long (torch.int64).");
  }
  if (csd_delta_pairs.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'csd_delta_pairs' to be of type long (torch.int64).");
  }
  if (csd_delta_counter.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'csd_delta_counter' to be of type int (torch.int32).");
  }
  if (csd_lookup_hit_ct.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'csd_lookup_hit_ct' to be of type long (torch.int64).");
  }
  if (csd_forced_accept_ct.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'csd_forced_accept_ct' to be of type long (torch.int64).");
  }
  if (csd_delta_pair_ct.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'csd_delta_pair_ct' to be of type long (torch.int64).");
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(batch_size);
  dim3 block(1);

  VerifyTreeGreedy<float, int32_t, int64_t><<<grid, block, 0, stream>>>(
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<int64_t*>(candidates.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<int64_t*>(target_predict.data_ptr()),
      static_cast<float*>(target_logits.data_ptr()),
      batch_size,
      num_spec_step,
      num_draft_tokens,
      vocab_size,
      static_cast<int64_t*>(csd_table_keys.data_ptr()),
      static_cast<int64_t*>(csd_delta_pairs.data_ptr()),
      static_cast<int32_t*>(csd_delta_counter.data_ptr()),
      static_cast<int64_t*>(csd_lookup_hit_ct.data_ptr()),
      static_cast<int64_t*>(csd_forced_accept_ct.data_ptr()),
      static_cast<int64_t*>(csd_delta_pair_ct.data_ptr()),
      static_cast<uint32_t>(csd_table_capacity),
      static_cast<uint32_t>(csd_table_max_probe),
      static_cast<uint32_t>(csd_delta_capacity),
      csd_enabled,
      csd_dynamic_update,
      csd_dynamic_update_ignore_prob_ratio,
      csd_force_accept_disabled,
      static_cast<float>(csd_logit_margin));
}
