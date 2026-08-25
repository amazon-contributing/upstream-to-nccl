/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA device handle struct, mirroring the layout defined in
 * aws-ofi-nccl (nccl_ofi_gin_gdaki_dev.h). The plugin's createContext()
 * populates this struct in GPU memory; the kernel code reads it.
 *
 * IMPORTANT: Must stay in sync with the plugin-side definition.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_

#include <stdint.h>

/*
 * Common per-endpoint state shared by every endpoint flavor. Holds
 * the GPU-resident QP, the target addressing table, and the per-QP
 * submitted and completed counts.
 */
/* The host CQ progress pass publishes per-peer completion state here, one entry
 * per CONTEXT per peer.
 *
 * ordered_completed_count_per_peer is a contiguous prefix in that peer's own
 * posting sequence; it pairs with submitted_count_per_peer[p] as its snapshot
 * ticket. error_at is 0 for the clear case, otherwise the failed pseq plus 1. The
 * bias is required because these are rolling uint32 compared as (int32_t)(a - b),
 * so a ~0u sentinel would fake an error on every wait.
 *
 * The host writes both words as ONE 8-byte store, so a waiter released by the
 * ordered prefix always sees the error belonging to that range.
 *
 * Attribution uses a (peer, pseq) split of req_id (peer in the high bits, pseq in
 * the low NCCL_OFI_GDAKI_PSEQ_BITS bits), echoed by efa_io_cdesc_common::req_id,
 * so the host reads peer and pseq straight from the CQE.
 *
 * Layout is shared with the plugin definition in aws-ofi-nccl
 * (nccl_ofi_gin_gdaki_dev.h) — keep them in sync. */
struct nccl_ofi_gin_gdaki_peer_completion {
  uint32_t ordered_completed_count_per_peer;
  uint32_t error_at;
};

struct nccl_ofi_gin_gdaki_dev_endpoint_handle {
  void* qp;                        /* GPU-resident QP (efa_cuda_qp layout) */

  /* Target addressing for this (poster) endpoint's QP.
   *
   * One GPU-resident table, sized [total_slots * nranks] and laid out
   * targetSlot-major: idx = targetSlot * nranks + peer, where
   *     targetSlot 0       -> peer's DATA endpoint
   *     targetSlot 1 + s   -> peer's sc endpoint s (signal id s)
   * and total_slots = 1 + (max over peers of their sc-endpoint count).
   *
   * The device selects the slot per write:
   *     plain put / counter-only write -> slot 0 (peer data EP, no
   *       FI_REMOTE_WRITE bound, so it ticks the local FI_WRITE counter
   *       without firing a signal on the receiver)
   *     signalling write (signal id s) -> slot 1 + s (peer sc EP s,
   *       whose FI_REMOTE_WRITE the GIN waitSignal observes)
   * The local poster QP is chosen by counterId (which endpoint owns this
   * handle); the remote target QP by the slot. Every (slot, peer) tuple
   * is resolved through THIS endpoint's own AV, so the data endpoint and
   * every sc endpoint each carry their own table. The stride (nranks)
   * lives on the top-level dev_handle.
   *
   * Layout is shared with the plugin definition in aws-ofi-nccl
   * (nccl_ofi_gin_gdaki_dev.h) — keep them in sync. */
  uint16_t* target_address_handles; /* [total_slots * nranks] */
  uint16_t* target_remote_qpns;     /* [total_slots * nranks] */
  uint32_t* target_qkey;            /* [total_slots * nranks] */

  /* This is the per-QP completed count: the endpoint's FI_WRITE NIC counter, one
   * increment per write completion. It suffices for ring reuse because the NIC
   * consumes WQEs in order and a completion implies its WQE was consumed. The
   * counter wraps at 2^31, so callers compare it under EFA_CNTR_MASK. */
  uint64_t* completed_count;

  /* `submitted_count` is incremented in ringDoorbell (an atomic add of the
   * newly-doorbelled span, by the group leader). (submitted_count -
   * *completed_count) is the number of WRs still in flight on this QP, used by
   * the SQ-overflow backpressure check and by Flush. */
  uint64_t submitted_count;

  /* This is the SQ ring size for this endpoint's QP. Put uses it to gate new
   * batches against in-flight WRs (efa-dp-direct's start_sq_batch leaves
   * ring-overflow checks to the caller). The kernel spins until
   * (submitted_count - *completed_count + batch_size) <= sq_size before reserving
   * slots. */
  uint32_t sq_size;

  uint32_t putvalue_pad;

  /* Base address of the PutValue source-slot pool for this endpoint. */
  uint64_t putvalue_slice_base;
};

/*
 * Per-signal/counter endpoint handle, returned to device code through
 * dev_handle->signal_handles[] and dev_handle->counter_handles[].
 *
 * Composes nccl_ofi_gin_gdaki_dev_endpoint_handle (qp / addressing /
 * counter completion tracking) and adds the cntr_value
 * pointer that the kernel reads to observe signal arrivals
 * (FI_REMOTE_WRITE) or counter increments (FI_WRITE).
 */
struct nccl_ofi_gin_gdaki_dev_counter_handle {
  /* Endpoint-common fields (qp, addressing,
   * counter completion tracking). */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle base;
  /* NIC writes the hardware counter value here (GPU memory). Read
   * via system-scope atomic load (hwCounterLoad / waitSignal). */
  uint64_t* cntr_value;

  /* Reset baseline for offset-based (reset-without-zeroing) semantics.
   * The NIC counter cannot be written by software, so ResetSignal /
   * ResetCounter snapshot cntr_value into cntr_offset instead of zeroing
   * the counter; reads/waits subtract cntr_offset. Initialized to 0 by
   * the plugin at populate() time. Must stay in sync with the plugin
   * definition in nccl_ofi_gin_gdaki_dev.h. */
  uint64_t cntr_offset;
};

/*
 * Device-visible handle returned from createContext, populated in GPU
 * memory by the plugin and dereferenced by the kernel.
 */
struct nccl_ofi_gin_gdaki_dev_handle {
  /* Data endpoint. Per-QP completion is its FI_WRITE NIC counter (completed_count). */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle data;

  /* Dedicated PutValue poster endpoint. */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle pvdata;

  /* Per-counter / per-signal endpoint handles, populated when the
   * caller asked createContext for nCounters / nSignals > 0. Both
   * arrays index into the same underlying signal/counter endpoint
   * (sc_endpoint) on the plugin side; they expose two views of that
   * endpoint with cntr_value pointing at the FI_WRITE counter
   * (counter_handles) or the FI_REMOTE_WRITE counter (signal_handles).
   * The array pointer is NULL when the corresponding count is zero. */
  struct nccl_ofi_gin_gdaki_dev_counter_handle** counter_handles; /* [nCounters] or NULL */
  struct nccl_ofi_gin_gdaki_dev_counter_handle** signal_handles;  /* [nSignals]  or NULL */
  int32_t nCounters;
  int32_t nSignals;

  int32_t nranks;
  int32_t rank;

  /* Multi-rail: the rail (EFA NIC) this logical context is bound to.
   * The plugin opens this context's endpoints on rail rail_id's domain
   * and bakes that rail's scratch MR keys into this handle. The kernel
   * uses rail_id only to index the per-rail memory handle array that
   * regMrSym returns as the window (see Put); all endpoint / counter /
   * scratch fields here are already rail-resolved, so the rest of the
   * device path is rail-agnostic.
   *
   * The plugin sets rail_id = contextId % num_rails, so on a
   * 2-NIC-per-GPU node distinct contextIds spread across both NICs. */
  uint32_t rail_id;

  /* Per-context signal-only scratch buffer, used by Put when the
   * caller has no payload (hasWins=false || bytes=0) but has
   * requested a signal/counter. EFA's RDMA write needs a real
   * remote address to bump the receiver's FI_REMOTE_WRITE counter,
   * so the plugin allocates a small buffer per createContext,
   * registers it on the proxy domain, and allgathers the
   * (local_addr, rkey) per rank. The kernel posts a 0-byte RDMA
   * write to the peer's scratch on the signal endpoint; the buffer
   * content is never read.
   *
   * scratch_remote_addrs / scratch_remote_rkeys are [nranks] arrays
   * in GPU memory.
   */
  uint32_t scratch_lkey;
  uint32_t scratch_pad;
  uint64_t scratch_local_addr;
  uint64_t* scratch_remote_addrs;
  uint32_t* scratch_remote_rkeys;

  /* lkey and slot stride for the pvdata PutValue source-slot pool (single
   * MR over the whole pool, uniform slot size). */
  uint32_t putvalue_lkey;
  uint32_t putvalue_slot_size;

  /* submitted_count_per_peer[p] is device-owned, sized [nranks]: it counts the
   *   writes to peer p that this CONTEXT doorbells across every QP. The post
   *   path's atomicAdd returns the write's position (pseq), which the code stamps
   *   into the low 8 bits of req_id; FlushAsync snapshots the counter as its
   *   ticket.
   * peer_completion[p] is host-published, sized [nranks]: it holds that peer's
   *   contiguous ordered_completed_count_per_peer plus its error position. Wait
   *   compares its ticket against this one word, so it needs only a single word
   *   per peer. */
  uint32_t* submitted_count_per_peer;
  nccl_ofi_gin_gdaki_peer_completion* peer_completion;

  /* This is the per-peer outstanding cap, in writes: put refuses to create the
   * (peer_window + 1)-th outstanding write to any single peer. This cap bounds the
   * host's per-peer bitmap. With 8-bit pseq the cap is 256, so pseq mod 256
   * uniquely identifies each live write. The value is a power of two. */
  uint32_t peer_window;

  /* Context-wide CQ-overflow gate: all QPs share one CQ, so unread CQEs summed
   * across every QP must stay within cq_depth. Device bumps
   * submitted_count_per_ctx per WQE; host publishes completed_count_per_ctx (one
   * per CQE read); put blocks while (submitted - completed) >= cq_depth. */
  uint64_t* submitted_count_per_ctx;
  uint64_t* completed_count_per_ctx;
  uint32_t cq_depth;
};

/*
 * Per-peer MR metadata. EFA's domain advertises FI_MR_VIRT_ADDR, so
 * RDMA write WQEs take absolute virtual addresses for both local and
 * remote buffers. The kernel passes an offset (srcOff / dstOff) and
 * the absolute address is computed as base_va + offset.
 */
struct nccl_ofi_gin_gdaki_mr_peer {
  uint64_t remote_addr;  /* remote rank's base VA for this MR */
  uint32_t rkey;         /* remote rank's rkey */
  uint32_t pad;
};

struct nccl_ofi_gin_gdaki_mr_handle {
  uint32_t lkey;
  int32_t nranks;
  uint64_t local_addr;                       /* local base VA for this MR */
  struct nccl_ofi_gin_gdaki_mr_peer peers[]; /* [nranks] flex array */
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_ */
