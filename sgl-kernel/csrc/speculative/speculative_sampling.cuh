/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2024-2025 by FlashInfer team.
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
#ifndef SPECULATIVE_SAMPLING_CUH_
#define SPECULATIVE_SAMPLING_CUH_

#include <assert.h>
#include <float.h>

#include <flashinfer/sampling.cuh>

namespace flashinfer {

namespace sampling {

using namespace cub;

static constexpr int64_t CSD_EMPTY_KEY = -1;

template <typename T>
struct CsdMaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

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

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    bool COMPUTE_MAX_LOGIT,
    typename DType,
    typename IdType>
__device__ __forceinline__ IdType CsdSampleResidualTokenAndMaxLogit(
    DType* target_probs,
    DType* draft_probs,
    DType* target_logits,
    uint32_t cur_prob_offset,
    uint32_t d,
    DType coin,
    bool load_draft_probs,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>& temp_storage,
    DType* max_target_logit) {
  const uint32_t tx = threadIdx.x;
  DType sum_relu_q_minus_p(0);
  DType thread_max_target_logit = -FLT_MAX;
  vec_t<DType, VEC_SIZE> q_vec, p_vec;
  DType relu_q_minus_p[VEC_SIZE];

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (load_draft_probs) {
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }
    if constexpr (COMPUTE_MAX_LOGIT) {
      vec_t<DType, VEC_SIZE> logits_vec;
      logits_vec.fill(DType(-FLT_MAX));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.load(target_logits + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        thread_max_target_logit = max(thread_max_target_logit, logits_vec[j]);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], DType(0));
    }
    sum_relu_q_minus_p += BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                              .Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }

  DType block_max_target_logit = 0;
  if constexpr (COMPUTE_MAX_LOGIT) {
    block_max_target_logit = BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                                 .Reduce(thread_max_target_logit, CsdMaxOp<DType>());
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
    if constexpr (COMPUTE_MAX_LOGIT) {
      *max_target_logit = block_max_target_logit;
    }
  }
  temp_storage.sampled_id = d - 1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  DType u = coin * sum_relu_q_minus_p;

  DType aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (load_draft_probs) {
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    vec_t<DType, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], DType(0));
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        i, d, [&](DType x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  return static_cast<IdType>(temp_storage.sampled_id);
}

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    typename DType>
__device__ __forceinline__ DType CsdComputeTargetEntropy(
    DType* target_probs,
    uint32_t cur_prob_offset,
    uint32_t d,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>& temp_storage) {
  const uint32_t tx = threadIdx.x;
  DType thread_entropy = 0;
  vec_t<DType, VEC_SIZE> q_vec;

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (q_vec[j] > DType(0)) {
        thread_entropy -= q_vec[j] * logf(q_vec[j]);
      }
    }
  }

  DType block_entropy = BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                            .Sum(thread_entropy);
  __syncthreads();
  return block_entropy;
}

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    bool CSD_ENABLED,
    bool CSD_DYNAMIC_UPDATE,
    typename DType,
    typename IdType,
    typename IdType2>
__global__ void TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,          // mutable
    IdType* accept_index,      // mutable
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    DType* target_logits,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc,
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
    DType csd_logit_margin,
    DType csd_force_accept_entropy_threshold) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;
  bool has_resampled_token = false;
  IdType2 resampled_token_id = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      DType target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      bool normal_accept = coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single;
      bool csd_force_accept = false;

      if (!normal_accept) {
        // FIXME: leverage draft probs
        draft_probs[cur_prob_offset + draft_token_id] = target_prob_single;

        if constexpr (CSD_ENABLED || CSD_DYNAMIC_UPDATE) {
          DType max_target_logit = 0;
          DType target_logit_single = target_logits[cur_prob_offset + draft_token_id];
          resampled_token_id = CsdSampleResidualTokenAndMaxLogit<
              BLOCK_THREADS,
              SCAN_ALGORITHM,
              REDUCE_ALGORITHM,
              VEC_SIZE,
              DETERMINISTIC,
              true,
              DType,
              IdType2>(
              target_probs,
              draft_probs,
              target_logits,
              cur_prob_offset,
              d,
              uniform_samples_for_final_sampling[bx],
              true,
              temp_storage,
              &max_target_logit);
          has_resampled_token = true;

          if (tx == 0) {
            int64_t csd_pair_key = CsdPackPair(draft_token_id, resampled_token_id);
            bool table_hit = CSD_ENABLED &&
                             CsdHashContains(csd_table_keys, csd_table_capacity, csd_table_max_probe, csd_pair_key);
            bool csd_logit_pass = target_logit_single >= max_target_logit + csd_logit_margin;
            if constexpr (CSD_DYNAMIC_UPDATE) {
              if (csd_dynamic_update_ignore_prob_ratio || csd_logit_pass) {
                CsdAppendDelta(csd_delta_pairs, csd_delta_counter, csd_delta_pair_ct, csd_delta_capacity, csd_pair_key);
              }
            }
            if (table_hit) {
              CsdAtomicAddI64(csd_lookup_hit_ct, 1ULL);
            }
            csd_force_accept = table_hit && csd_logit_pass && !csd_force_accept_disabled;
            if (csd_force_accept && csd_force_accept_entropy_threshold >= DType(0)) {
              temp_storage.block_aggregate.value = DType(2);
            } else {
              temp_storage.block_aggregate.value = csd_force_accept ? DType(1) : DType(0);
            }
          }
          __syncthreads();
          bool should_check_csd_entropy = temp_storage.block_aggregate.value == DType(2);
          __syncthreads();
          if (should_check_csd_entropy) {
            DType target_entropy = CsdComputeTargetEntropy<
                BLOCK_THREADS,
                SCAN_ALGORITHM,
                REDUCE_ALGORITHM,
                VEC_SIZE,
                DType>(target_probs, cur_prob_offset, d, temp_storage);
            if (tx == 0) {
              temp_storage.block_aggregate.value =
                  target_entropy <= csd_force_accept_entropy_threshold ? DType(1) : DType(0);
            }
            __syncthreads();
          }
          csd_force_accept = temp_storage.block_aggregate.value != DType(0);
          if (tx == 0 && csd_force_accept) {
            CsdAtomicAddI64(csd_forced_accept_ct, 1ULL);
          }
          __syncthreads();
        }
      }

      if (normal_accept || csd_force_accept) {
        // accept token
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        has_resampled_token = false;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;

  if (!has_resampled_token) {
    DType max_target_logit = 0;
    resampled_token_id = CsdSampleResidualTokenAndMaxLogit<
        BLOCK_THREADS,
        SCAN_ALGORITHM,
        REDUCE_ALGORITHM,
        VEC_SIZE,
        DETERMINISTIC,
        false,
        DType,
        IdType2>(
        target_probs,
        draft_probs,
        target_logits,
        cur_prob_offset,
        d,
        uniform_samples_for_final_sampling[bx],
        num_accepted_tokens != num_speculative_tokens - 1,
        temp_storage,
        &max_target_logit);
  }
  predicts[last_accepted_retrive_idx] = resampled_token_id;
  // value at not used indices are undefined
}

template <
    uint32_t BLOCK_THREADS,
    bool CSD_ENABLED,
    bool CSD_DYNAMIC_UPDATE,
    typename DType,
    typename IdType,
    typename IdType2>
cudaError_t LaunchTreeSpeculativeSamplingTargetOnly(
    IdType* predicts,
    IdType* output_token_ids,
    IdType* output_accepted_token_num,
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    DType* target_logits,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc,
    bool deterministic,
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
    DType csd_logit_margin,
    DType csd_force_accept_entropy_threshold,
    cudaStream_t stream) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);
  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  float capped_threshold_acc = fmaxf(threshold_acc, 1e-9f);
  void* args[] = {
      &predicts,
      &output_token_ids,
      &output_accepted_token_num,
      &candidates,
      &retrive_index,
      &retrive_next_token,
      &retrive_next_sibling,
      &uniform_samples,
      &uniform_samples_for_final_sampling,
      &target_probs,
      &draft_probs,
      &target_logits,
      &batch_size,
      &num_speculative_tokens,
      &num_draft_tokens,
      &d,
      &threshold_single,
      &capped_threshold_acc,
      &csd_table_keys,
      &csd_delta_pairs,
      &csd_delta_counter,
      &csd_lookup_hit_ct,
      &csd_forced_accept_ct,
      &csd_delta_pair_ct,
      &csd_table_capacity,
      &csd_table_max_probe,
      &csd_delta_capacity,
      &csd_enabled,
      &csd_dynamic_update,
      &csd_dynamic_update_ignore_prob_ratio,
      &csd_force_accept_disabled,
      &csd_logit_margin,
      &csd_force_accept_entropy_threshold};
  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        auto kernel = TreeSpeculativeSamplingTargetOnly<
            BLOCK_THREADS,
            SCAN_ALGO,
            REDUCE_ALGO,
            VEC_SIZE,
            DETERMINISTIC,
            CSD_ENABLED,
            CSD_DYNAMIC_UPDATE,
            DType,
            IdType,
            IdType2>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      })});
  return cudaSuccess;
}

template <typename DType, typename IdType, typename IdType2>
cudaError_t TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,                   // mutable
    IdType* output_token_ids,           // mutable
    IdType* output_accepted_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    DType* target_logits,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single = 1,
    DType threshold_acc = 1,
    bool deterministic = true,
    const int64_t* csd_table_keys = nullptr,
    int64_t* csd_delta_pairs = nullptr,
    int32_t* csd_delta_counter = nullptr,
    int64_t* csd_lookup_hit_ct = nullptr,
    int64_t* csd_forced_accept_ct = nullptr,
    int64_t* csd_delta_pair_ct = nullptr,
    uint32_t csd_table_capacity = 0,
    uint32_t csd_table_max_probe = 0,
    uint32_t csd_delta_capacity = 0,
    bool csd_enabled = false,
    bool csd_dynamic_update = false,
    bool csd_dynamic_update_ignore_prob_ratio = false,
    bool csd_force_accept_disabled = false,
    DType csd_logit_margin = -4.605170185988091f,
    DType csd_force_accept_entropy_threshold = -1.0f,
    cudaStream_t stream = 0) {
  if (csd_enabled && csd_dynamic_update) {
    return LaunchTreeSpeculativeSamplingTargetOnly<512, true, true>(
        predicts,
        output_token_ids,
        output_accepted_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        target_logits,
        batch_size,
        num_speculative_tokens,
        num_draft_tokens,
        d,
        threshold_single,
        threshold_acc,
        deterministic,
        csd_table_keys,
        csd_delta_pairs,
        csd_delta_counter,
        csd_lookup_hit_ct,
        csd_forced_accept_ct,
        csd_delta_pair_ct,
        csd_table_capacity,
        csd_table_max_probe,
        csd_delta_capacity,
        csd_enabled,
        csd_dynamic_update,
        csd_dynamic_update_ignore_prob_ratio,
        csd_force_accept_disabled,
        csd_logit_margin,
        csd_force_accept_entropy_threshold,
        stream);
  }
  if (csd_enabled) {
    return LaunchTreeSpeculativeSamplingTargetOnly<512, true, false>(
        predicts,
        output_token_ids,
        output_accepted_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        target_logits,
        batch_size,
        num_speculative_tokens,
        num_draft_tokens,
        d,
        threshold_single,
        threshold_acc,
        deterministic,
        csd_table_keys,
        csd_delta_pairs,
        csd_delta_counter,
        csd_lookup_hit_ct,
        csd_forced_accept_ct,
        csd_delta_pair_ct,
        csd_table_capacity,
        csd_table_max_probe,
        csd_delta_capacity,
        csd_enabled,
        csd_dynamic_update,
        csd_dynamic_update_ignore_prob_ratio,
        csd_force_accept_disabled,
        csd_logit_margin,
        csd_force_accept_entropy_threshold,
        stream);
  }
  if (csd_dynamic_update) {
    return LaunchTreeSpeculativeSamplingTargetOnly<512, false, true>(
        predicts,
        output_token_ids,
        output_accepted_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        target_logits,
        batch_size,
        num_speculative_tokens,
        num_draft_tokens,
        d,
        threshold_single,
        threshold_acc,
        deterministic,
        csd_table_keys,
        csd_delta_pairs,
        csd_delta_counter,
        csd_lookup_hit_ct,
        csd_forced_accept_ct,
        csd_delta_pair_ct,
        csd_table_capacity,
        csd_table_max_probe,
        csd_delta_capacity,
        csd_enabled,
        csd_dynamic_update,
        csd_dynamic_update_ignore_prob_ratio,
        csd_force_accept_disabled,
        csd_logit_margin,
        csd_force_accept_entropy_threshold,
        stream);
  }
  return LaunchTreeSpeculativeSamplingTargetOnly<1024, false, false>(
      predicts,
      output_token_ids,
      output_accepted_token_num,
      candidates,
      retrive_index,
      retrive_next_token,
      retrive_next_sibling,
      uniform_samples,
      uniform_samples_for_final_sampling,
      target_probs,
      draft_probs,
      target_logits,
      batch_size,
      num_speculative_tokens,
      num_draft_tokens,
      d,
      threshold_single,
      threshold_acc,
      deterministic,
      csd_table_keys,
      csd_delta_pairs,
      csd_delta_counter,
      csd_lookup_hit_ct,
      csd_forced_accept_ct,
      csd_delta_pair_ct,
      csd_table_capacity,
      csd_table_max_probe,
      csd_delta_capacity,
      csd_enabled,
      csd_dynamic_update,
      csd_dynamic_update_ignore_prob_ratio,
      csd_force_accept_disabled,
      csd_logit_margin,
      csd_force_accept_entropy_threshold,
      stream);
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // SPECULATIVE_SAMPLING_CUH_
