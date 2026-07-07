/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA implementations for NCCL GIN device-side APIs.
 *
 * This file provides ncclGinApi_*<NCCL_NET_DEVICE_GIN_EFA_GDA> template
 * specializations that target EFA via efa-dp-direct.
 *
 * Implemented: Put (data + signal/counter endpoints, signal-only via
 *              scratch buffer), Flush, GetSignalPtr, GetCounterPtr,
 *              ResetSignal, ResetCounter.
 * Stub: PutValue, Get, FlushAsync, Wait.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_H_

#include <cstdint>
#include <cuda/atomic>
#include <cooperative_groups.h>

#include "../gin_device_common.h"
#include "gin_efa_gda_dev.h"

/* efa-dp-direct device functions (inline implementations) */
#include "../../transport/net_efa_gda/efa-dp-direct/include/device/efa_cuda_dp_impl.cuh"

namespace nccl {
namespace gin {
namespace efa_gda {

/* The plugin returns a contiguous array of per-context dev handles
 * in GPU memory; ctx.handle points at element 0. ctx.contextId
 * selects the entry for this caller. */
NCCL_DEVICE_INLINE static nccl_ofi_gin_gdaki_dev_handle*
getDevHandle(ncclGinCtx ctx) {
  return &((nccl_ofi_gin_gdaki_dev_handle*)ctx.handle)[ctx.contextId];
}

/* ── Mode mapping: NCCL → efa-dp-direct ───────────────────────────── */

template <ncclGinResourceSharingMode mode>
static constexpr cuda::thread_scope ncclGinScope =
    (mode == NCCL_GIN_RESOURCE_SHARING_CTA)
        ? cuda::thread_scope_block : cuda::thread_scope_device;

/* The EFA hardware completion counters (FI_WRITE / FI_REMOTE_WRITE) wrap at
 * 2^31, while the kernel-side producer cursors are uint32 (wrap at 2^32).
 * Every comparison between a producer cursor and a HW counter must therefore
 * be a modular difference reduced to 31 bits: compute (producer - consumer)
 * and mask with EFA_CNTR_MASK. The true difference (in-flight / outstanding
 * work) is always far below 2^31 (bounded by sq_size == 4096), so the masked
 * difference recovers the exact value regardless of how many times either side
 * has wrapped. Never compare absolute counter values. */
static constexpr uint32_t EFA_CNTR_MASK = 0x7fffffffu;

/* ── Atomic primitives parameterized on scope and memory order ────── */

template <cuda::thread_scope Scope, cuda::memory_order Order>
NCCL_DEVICE_INLINE static uint64_t scopedAtomicLoad(uint64_t *ptr) {
  cuda::atomic_ref<uint64_t, Scope> r(*ptr);
  return r.load(Order);
}

template <cuda::thread_scope Scope, cuda::memory_order Order>
NCCL_DEVICE_INLINE static void scopedAtomicAdd(uint64_t *ptr, uint64_t val) {
  cuda::atomic_ref<uint64_t, Scope> r(*ptr);
  r.fetch_add(val, Order);
}

/* ── NIC-written hardware counter (FI_WRITE / FI_REMOTE_WRITE) ────── */

/* Read a NIC-written hardware counter from GPU memory. Uses system-scope
 * acquire so the load is coherent with the NIC's PCIe writes (bypasses
 * GPU caches) and subsequent operations on this thread cannot be
 * reordered to before the load. The acquire matches libfabric's
 * local-completion contract: when this load observes the counter has
 * reached a target, the NIC's prior side effects (e.g. source-buffer
 * DMA-reads complete) are ordered-before whatever this thread does
 * next (e.g. overwriting that source buffer or reusing the slot). */
NCCL_DEVICE_INLINE static uint64_t hwCounterLoad(uint64_t *ptr) {
  return scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_acquire>(ptr);
}

/* ── postRdmaWrite: shared post path for Put and PutValue ─────────── */

/* Posts an RDMA write on `ep`'s local QP (its FI_WRITE counter tracks
 * local completion) to the remote QP given by the explicit
 * (ah, qpn, qkey) tuple. The local poster QP and the remote target QP
 * are chosen independently by the caller: counterId selects the local
 * poster (this `ep`), the target slot selects the remote tuple (via the
 * poster's [total_slots*nranks] target table). */
template <ncclGinResourceSharingMode mode>
NCCL_DEVICE_INLINE static void postRdmaWrite(
    nccl_ofi_gin_gdaki_dev_endpoint_handle *ep, uint16_t ah, uint16_t qpn,
    uint32_t qkey, uint64_t srcAddr, uint32_t srcLkey, uint32_t writeBytes,
    uint64_t dstAddr, uint32_t dstRkey) {

  efa_cuda_qp       *qp                  = (efa_cuda_qp *)ep->qp;
  uint64_t          *submitted_count_ptr  = &ep->submitted_count;
  uint64_t          *local_cntr_ptr       = ep->local_cntr_value;
  uint32_t           sq_size_val          = ep->sq_size;

  efa_io_tx_wqe wr;
  efa_cuda_init_rdma_write_wr(&wr, (uint16_t)threadIdx.x, dstRkey, dstAddr);
  efa_cuda_wr_set_sge(&wr, srcLkey, srcAddr, writeBytes);
  efa_cuda_wr_set_remote(&wr, ah, (uint32_t)qpn, qkey);

  /* Sliding-window SQ post with warp coalescing (Stage 2).
   *
   * Inlines the reserve / write / doorbell sequence directly against
   * the efa_cuda_qp ring (uses only the WQE *builders* above, not the
   * efa-dp-direct start_sq_batch / sq_batch_place_wr / flush_sq_wrs
   * helpers). Two shared cursors in the QP coordinate all posters
   * (across lanes, warps and CTAs):
   *
   *   pc             : monotonic reservation index. A group's leader
   *                    claims its whole range with one atomicAdd(+g).
   *   wqes_completed : "released" cursor — the doorbell has been rung
   *                    up to here. Doubles as the sliding-window base
   *                    and the doorbell-order rendezvous token.
   *
   * Coalescing: lanes of a warp targeting the same QP form a group via
   * coalesced_threads() + labeled_partition(qp). The leader reserves g
   * = group.num_threads() contiguous slots; every member writes its own
   * WQE in parallel; the leader rings one doorbell for the batch.
   *
   * max_batch bound: a group may be larger than the EFA staging limit
   * (a warp can have up to 32 lanes on one QP), so the group is chunked
   * into windows of <= max_batch. For each chunk:
   *   - window-wait (leader): write only once the chunk fits within the
   *     released window [released, released + max_batch). This bounds
   *     un-doorbelled WQEs across ALL concurrent groups to max_batch.
   *   - SQ ring-overflow wait (leader): the chunk's high-water slot must
   *     be within sq_size of the NIC consumer (FI_WRITE counter).
   *   - members write their WQEs in parallel.
   *   - doorbell rendezvous (leader): wait until released == chunk_base
   *     (strict slot order across groups), ring the doorbell, then
   *     advance released to hand off to the next group. */
  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  auto group = cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)qp);

  int  my_idx     = group.thread_rank();
  int  group_size = group.num_threads();
  bool is_leader  = (my_idx == 0);
  uint32_t max_batch = qp->sq.wq.max_batch;

  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> pc_ref(qp->sq.wq.pc);
  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> base_ref(qp->sq.wq.wqes_completed);

  /* Leader reserves the whole group's contiguous slot range. */
  uint32_t base = 0;
  if (is_leader) {
    base = pc_ref.fetch_add((uint32_t)group_size, cuda::memory_order_relaxed);
  }
  base = group.shfl(base, 0);

  /* Chunk the group into windows of <= max_batch. */
  for (int chunk_start = 0; chunk_start < group_size; chunk_start += (int)max_batch) {
    int      chunk_size = min((int)max_batch, group_size - chunk_start);
    uint32_t chunk_base = base + (uint32_t)chunk_start;
    uint32_t chunk_next = chunk_base + (uint32_t)chunk_size;

    if (is_leader) {
      /* Sliding-window backpressure: keep cumulative un-doorbelled
       * WQEs (across all groups) within max_batch. */
      while (chunk_next > base_ref.load(cuda::memory_order_acquire) + max_batch) {
        /* spin */
      }
      /* SQ ring-overflow backpressure on the chunk's high-water slot.
       * System-scope acquire so we see the latest NIC FI_WRITE update
       * and the WQE stores below can't hoist above this load.
       *
       * In-flight count is computed as a 31-bit modular difference
       * (producer chunk_next minus the NIC FI_WRITE counter): the HW
       * counter wraps at 2^31 and chunk_next is uint32, so a plain
       * widened subtraction would underflow once either side wraps.
       * The true in-flight depth is bounded by sq_size (4096) « 2^31,
       * so the masked difference is exact. */
      while (((chunk_next - (uint32_t)hwCounterLoad(local_cntr_ptr)) & EFA_CNTR_MASK) > sq_size_val) {
        /* spin */
      }
    }
    group.sync();   /* members wait for leader's backpressure before writing */

    /* Members in this window write their own WQE into their slot. */
    if (my_idx >= chunk_start && my_idx < chunk_start + chunk_size) {
      uint32_t my_slot = chunk_base + (uint32_t)(my_idx - chunk_start);
      uint32_t sq_idx  = my_slot & qp->sq.wq.queue_mask;
      int wqe_phase    = (int)((my_slot >> qp->sq.wq.queue_size_shift) & 1u);
      EFA_SET(&wr.meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase);
      uint64_t *src = (uint64_t *)&wr;
      uint64_t *dst = (uint64_t *)(qp->sq.wq.buf + sq_idx * sizeof(efa_io_tx_wqe));
      for (int i = 0; i < 8; i++)
        dst[i] = src[i];
    }
    group.sync();   /* all members' WQE writes for this chunk are done */

    if (is_leader) {
      __threadfence_system();   /* publish all chunk WQE writes before the doorbell */
      /* Doorbell-order rendezvous: ring in slot order across groups. */
      while (base_ref.load(cuda::memory_order_acquire) != chunk_base) {
        /* spin */
      }
      *qp->sq.wq.db = chunk_next;
      __threadfence_system();   /* drain/order the doorbell write */
      scopedAtomicAdd<ncclGinScope<mode>, cuda::memory_order_relaxed>(submitted_count_ptr, (uint64_t)chunk_size);
      base_ref.store(chunk_next, cuda::memory_order_release);   /* hand off to next group */
    }
    group.sync();   /* chunk fully posted before the next chunk */
  }
}

/* ── putImplMode: mode-templated Put implementation ─────────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                    ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                    size_t srcOff, size_t bytes,
                                    ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                    uint64_t signalOpArg, bool hasCounter,
                                    ncclGinCounter_t counterId, bool hasDescriptor,
                                    ncclGinDescriptorSmem* descriptor,
                                    cuda::thread_scope required, cuda::thread_scope given,
                                    uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle *dev = getDevHandle(ctx);

    bool hasPayload    = hasWins && bytes > 0;
    bool needsSignalEp = (signal.type != NCCL_GIN_SIGNAL_TYPE_NONE) || hasCounter;

    /* This backend supports INDEXED signals only. EFA's FI_REMOTE_WRITE
     * counter ticks exactly once per inbound write and has no atomic-add,
     * so a signal Add-by-N is emulated as N inbound write events (see the
     * posting block below). VA-typed signals are not representable. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE
            || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
           && "EFA GDA: only INDEXED signals are supported");
    assert((signal.type != NCCL_GIN_SIGNAL_TYPE_INDEXED
            || (int)signal.indexedSignal.signalId < dev->nSignals)
           && "EFA GDA: signalId out of range");
    assert((!hasCounter || (int)counterId < dev->nCounters)
           && "EFA GDA: counterId out of range");

    if (hasPayload || needsSignalEp) {
      /* Two WQE patterns:
       *
       * (a) Data put: posts an RDMA write of the user payload.
       *     Routed through signal/counter endpoint when needsSignalEp
       *     so the receiver's FI_REMOTE_WRITE fires on completion;
       *     otherwise routed through the data endpoint.
       *
       * (b) Signal-only: posts a 0-byte RDMA write into the peer's
       *     per-context scratch buffer. The write event bumps the
       *     receiver's FI_REMOTE_WRITE counter on the signal endpoint. */
      uint64_t absSrcAddr;
      uint64_t absDstAddr;
      uint32_t dstRkey;
      uint32_t srcLkey;
      uint32_t writeBytes;
      if (hasPayload) {
        nccl_ofi_gin_gdaki_mr_handle *dstMh = (nccl_ofi_gin_gdaki_mr_handle *)dstWin;
        nccl_ofi_gin_gdaki_mr_handle *srcMh = (nccl_ofi_gin_gdaki_mr_handle *)srcWin;
        absSrcAddr = srcMh->local_addr + srcOff;
        absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
        dstRkey    = dstMh->peers[peer].rkey;
        srcLkey    = srcMh->lkey;
        writeBytes = (uint32_t)bytes;
      } else {
        absSrcAddr = dev->scratch_local_addr;
        absDstAddr = dev->scratch_remote_addrs[peer];
        dstRkey    = dev->scratch_remote_rkeys[peer];
        srcLkey    = dev->scratch_lkey;
        writeBytes = 0;
      }

      /* A single RDMA write carries two independent QP choices:
       *
       *   - Local poster QP: the SQ we post from; its FI_WRITE counter
       *     ticks on local completion. This is a LOCAL property,
       *     selected by counterId. With a counter request we post from
       *     counter_handles[counterId] so its FI_WRITE is the counter;
       *     otherwise we post from the data endpoint (a signal-only or
       *     plain put has no local counter to track here).
       *
       *   - Remote target QP: the peer endpoint we address; the peer's
       *     FI_REMOTE_WRITE counter on THAT endpoint ticks (only when the
       *     target is a signal/sc endpoint). This is a TARGET property,
       *     selected below as a slot into the poster's target
       *     table (slot 0 = peer data EP, slot 1+signalId = peer sc EP).
       *
       * The signal (signalId) selects the remote (target) QP via the
       * slot; the counter (counterId) selects the local (poster) QP. */
      nccl_ofi_gin_gdaki_dev_endpoint_handle *main_ep =
          hasCounter ? &dev->counter_handles[counterId]->base : &dev->data;

      /* Target slot in the [total_slots*nranks] target addressing table
       * (targetSlot-major: idx = targetSlot*nranks + peer):
       *     signalling write (INDEXED) -> slot 1 + signalId (peer sc EP,
       *       whose FI_REMOTE_WRITE the receiver's waitSignal observes)
       *     plain put / counter-only    -> slot 0 (peer DATA EP, which
       *       binds no FI_REMOTE_WRITE, so the write ticks the local
       *       FI_WRITE counter without firing a signal on the receiver)
       * The local poster QP is chosen by counterId (main_ep); the slot
       * picks the remote target. */
      const bool isIndexed = (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED);
      const uint32_t targetSlot =
          isIndexed ? (1u + (uint32_t)signal.indexedSignal.signalId) : 0u;
      const uint32_t targetIdx = targetSlot * (uint32_t)dev->nranks + (uint32_t)peer;

      /* Signal increment count.
       *
       * EFA's FI_REMOTE_WRITE counter advances by exactly 1 per inbound
       * write, so an Add-by-N signal is emulated as N inbound write
       * events. Inc is always +1 (signalOpArg is defined to be 1 for Inc
       * by the GIN API). A pure data put or counter request (no signal)
       * contributes a single write.
       *
       * Correctness-first: this issues the writes as separate posts (one
       * doorbell each). A future optimization can batch the doorbell over
       * a larger reservation via postRdmaWrite's chunk loop; that must NOT
       * be done by suppressing doorbells across calls, which would break
       * the wqes_completed sliding-window / rendezvous invariant.
       *
       * TODO: batch the doorbells for a signal Add-by-N (and across the
       * payload + scratch writes) instead of ringing one doorbell per
       * write. Must reuse postRdmaWrite's chunk loop (one doorbell per
       * max_batch reservation), NOT a doorbell-suppress flag across
       * separate calls. */
      uint32_t signalCount = 1u;
      if (isIndexed && signalOp == ncclGinSignalAdd) {
        signalCount = (uint32_t)signalOpArg;
      }

      /* Target tuples, both read from the target table at the same slot.
       *
       * The local FI_WRITE counter selected by counterId must tick
       * EXACTLY ONCE per Put, no matter how many physical writes the
       * signal Add-by-N expands into. So only the FIRST write rides
       * `main_ep` (the counterId-selected poster when hasCounter, else
       * the data EP); every remaining (signalCount - 1) increment rides
       * the DATA endpoint, whose FI_WRITE is not the caller's counter.
       * Both resolve the SAME remote target (slot), but each through its
       * own endpoint's AV — an address handle is AV-local, so the data EP
       * uses its own tuple, not main_ep's. */
      const uint16_t main_ah   = main_ep->target_address_handles[targetIdx];
      const uint16_t main_qpn  = main_ep->target_remote_qpns[targetIdx];
      const uint32_t main_qkey = main_ep->target_qkey[targetIdx];
      const uint16_t dataSigAh   = dev->data.target_address_handles[targetIdx];
      const uint16_t dataSigQpn  = dev->data.target_remote_qpns[targetIdx];
      const uint32_t dataSigQkey = dev->data.target_qkey[targetIdx];

      /* First write: payload (hasPayload) or 0-byte scratch (signal-only),
       * on the counterId-selected poster so the local counter ticks once,
       * addressed to the resolved target so the receiver's FI_REMOTE_WRITE
       * fires once. absSrcAddr/absDstAddr/writeBytes already point at the
       * payload or the scratch region per the hasPayload branch above. */
      postRdmaWrite<mode>(main_ep, main_ah, main_qpn, main_qkey, absSrcAddr, srcLkey,
                          writeBytes, absDstAddr, dstRkey);

      /* Remaining (signalCount - 1) signal increments: 0-byte writes to
       * the peer scratch region on the DATA endpoint, so the caller's
       * counter is not over-incremented. The loop body is empty unless
       * signalCount > 1, which implies an INDEXED Add (and thus a
       * signal endpoint target). */
      for (uint32_t k = 1u; k < signalCount; k++) {
        postRdmaWrite<mode>(&dev->data, dataSigAh, dataSigQpn, dataSigQkey,
                            dev->scratch_local_addr, dev->scratch_lkey, 0u,
                            dev->scratch_remote_addrs[peer],
                            dev->scratch_remote_rkeys[peer]);
      }
    }
  }
  (void)hasDescriptor; (void)descriptor;
  (void)required; (void)given; (void)optFlags;
  coop.sync();
}

/* ── putImpl: runtime mode dispatcher ─────────────────────────────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                    ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                    size_t srcOff, size_t bytes,
                                    ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                    uint64_t signalOpArg, bool hasCounter,
                                    ncclGinCounter_t counterId, bool hasDescriptor,
                                    ncclGinDescriptorSmem* descriptor,
                                    cuda::thread_scope required, cuda::thread_scope given,
                                    uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      putImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        signal, signalOp, signalOpArg, hasCounter, counterId,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
    default:
      putImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        signal, signalOp, signalOpArg, hasCounter, counterId,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
  }
}

/* ── flushImplMode: mode-templated Flush implementation ───────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void flushImplMode(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
  (void)ord;
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle *dev = getDevHandle(ctx);

    /* For each endpoint with outstanding work, snapshot submitted_count
     * (scoped atomic load matching the relaxed bumps from the post
     * path), then spin on the NIC-written FI_WRITE counter until it
     * reaches the snapshot. The HW counter is read with system-scope
     * acquire so the GPU bypasses caches and observes the latest NIC
     * update through PCIe-coherent memory. */
    auto wait_for_endpoint = [abortFlag](nccl_ofi_gin_gdaki_dev_endpoint_handle &ep) -> bool {
      uint64_t target = scopedAtomicLoad<ncclGinScope<mode>, cuda::memory_order_relaxed>(&ep.submitted_count);

      /* Drain-to-zero: outstanding = (submitted - completed) reduced to
       * 31 bits, since the NIC FI_WRITE counter wraps at 2^31. Wait until
       * no work is outstanding. Outstanding is bounded by sq_size « 2^31,
       * so the masked difference is exact and cannot be fooled by a
       * counter wrap. */
      while (((((uint32_t)target) - (uint32_t)hwCounterLoad(ep.local_cntr_value)) & EFA_CNTR_MASK) != 0) {
        if (abortFlag && *abortFlag) return false;
      }
      return true;
    };

    if (!wait_for_endpoint(dev->data)) return;

    /* Drain the counter endpoints only. With the decoupled model the
     * local poster QP is always either the data endpoint or a counter
     * endpoint (counterId selects the poster); a signal endpoint is
     * only ever a remote TARGET, never a local poster, so its FI_WRITE
     * counter never ticks from our writes and there is nothing to
     * drain. A signal QP needs no local completions at all. */
    for (int i = 0; i < dev->nCounters; i++) {
      if (!wait_for_endpoint(dev->counter_handles[i]->base)) return;
    }
  }
  coop.sync();
}

/* ── flushImpl: runtime mode dispatcher ───────────────────────────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void flushImpl(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      flushImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, ord, abortFlag);
      break;
    default:
      flushImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, ord, abortFlag);
      break;
  }
}

} // namespace efa_gda
} // namespace gin
} // namespace nccl

/* ── Put ───────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    nccl::gin::efa_gda::putImpl(
      ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
      signal, signalOp, signalOpArg, hasCounter, counterId,
      hasDescriptor, descriptor, required, given, optFlags);
  }
};

/* ── PutValue ─────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    coop.sync();
    /* TODO: efa-dp-direct wr_set_inline_data only supports SEND opcode,
       not RDMA_WRITE. Need either an efa-dp-direct update or a
       pre-registered scratch buffer approach. */
    (void)ctx; (void)peer; (void)dstWin; (void)dstOff; (void)srcVal;
    (void)signal; (void)signalOp; (void)signalOpArg; (void)hasDescriptor;
    (void)descriptor; (void)required; (void)given; (void)optFlags;
    coop.sync();
  }
};

/* ── Get ──────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    coop.sync();
    /* TODO: implement with efa_cuda_init_rdma_read_wr */
    (void)ctx; (void)peer; (void)remoteWin; (void)remoteOff;
    (void)localWin; (void)localOff; (void)bytes;
    (void)hasDescriptor; (void)descriptor; (void)optFlags;
    coop.sync();
  }
};

/* ── FlushAsync ───────────────────────────────────────────────────── */

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    (void)ctx; (void)peer; (void)outRequest; (void)hasDescriptor; (void)descriptor; (void)optFlags;
  }
};

/* ── Wait ─────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)ctx; (void)request; (void)hasDescriptor;
    (void)descriptor; (void)ord; (void)abortFlag;
  }
};

/* ── Flush ────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor; (void)descriptor;
    nccl::gin::efa_gda::flushImpl(ctx, coop, ord, abortFlag);
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle *h = dev->signal_handles[signalId];
    return { (uint64_t *)h->cntr_value, h->cntr_offset };
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle *h = dev->counter_handles[counterId];
    return { (uint64_t *)h->cntr_value, h->cntr_offset };
  }
};

/* ── ResetSignal ──────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert(signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED
           && "EFA GDA ResetSignal: only INDEXED signals are supported");
    assert((int)signal.indexedSignal.signalId < dev->nSignals
           && "EFA GDA ResetSignal: signalId out of range");
    /* Offset-based reset: the NIC counter cannot be written, so snapshot
     * its current value into cntr_offset. Subsequent reads/waits subtract
     * the offset, making the signal appear reset. */
    nccl_ofi_gin_gdaki_dev_counter_handle *h =
        dev->signal_handles[signal.indexedSignal.signalId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t *)h->cntr_value);
  }
};

/* ── ResetCounter ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert((int)counterId < dev->nCounters
           && "EFA GDA ResetCounter: counterId out of range");
    /* Offset-based reset: snapshot the NIC counter into cntr_offset
     * instead of writing the (NIC-owned) counter. */
    nccl_ofi_gin_gdaki_dev_counter_handle *h = dev->counter_handles[counterId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t *)h->cntr_value);
  }
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_H_ */
