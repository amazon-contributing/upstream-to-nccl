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
 * the GPU-resident QP/CQ, per-peer addressing, the per-QP spinlock
 * that serializes the device-side WQE-post sequence, and the
 * counter-based completion tracking fields.
 */
struct nccl_ofi_gin_gdaki_dev_endpoint_handle {
  void *qp;                        /* GPU-resident QP (efa_cuda_qp layout) */
  void *cq;                        /* GPU-resident CQ (efa_cuda_cq layout) */
  uint16_t *address_handles;       /* [nranks] per-peer addressing */
  uint16_t *remote_qpns;
  uint32_t *qkey;

  /* Per-QP spinlock for the device-side WQE post path. efa-dp-direct's
   * start_sq_batch / sq_batch_place_wr / flush_sq_wrs sequence is
   * single-threaded per QP (per the efa-dp-direct CUDA README). One
   * lock per endpoint lets multiple CTAs targeting different endpoints
   * proceed in parallel; only CTAs targeting the same endpoint contend. */
  uint32_t sq_lock;

  /* Counter-based completion tracking.
   *
   * `local_cntr_value` points at the FI_WRITE hardware counter for this
   * endpoint's QP. The NIC increments it on every locally-completed
   * outgoing WR. The kernel reads it directly from GPU memory.
   *
   * `submitted_count` is incremented by the device under sq_lock after
   * a successful flush_sq_wrs. (submitted_count - *local_cntr_value)
   * gives the number of WRs still in flight on this QP — used by the
   * SQ-overflow backpressure check and by Flush to wait for local
   * completion. */
  /* `local_cntr_value` is read via cuda::atomic_ref with system scope
   * (see hwCounterLoad helper) since the NIC writes it via PCIe and
   * we need to bypass GPU caches when polling. */
  uint64_t *local_cntr_value;
  uint64_t submitted_count;

  /* SQ ring size for this endpoint's QP. Used by Put to gate new
   * batches against in-flight WRs (efa-dp-direct's start_sq_batch
   * does not validate ring overflow on its own). The kernel spins
   * until (submitted_count - *local_cntr_value + batch_size)
   * <= sq_size before reserving slots. */
  uint32_t sq_size;

  /* Signal-target addressing for this (poster) endpoint.
   *
   * Three GPU-resident arrays that let THIS endpoint post a
   * signalling RDMA write addressed to peer P's signal endpoint
   * number `signalId` — i.e. the write ticks the FI_REMOTE_WRITE
   * counter of the TARGET's sc endpoint `signalId`, not the
   * same-index endpoint as the local poster. This is what makes the
   * GIN signalId a property of the target, decoupled from which
   * local endpoint posts the write (the local poster's own FI_WRITE
   * counter is selected independently by counterId).
   *
   * Indexed signalId-major:
   *     idx = signalId * dev_handle->nranks + peer
   * with signalId in [0, nSignals) and peer in [0, nranks). The row
   * count (nSignals) and stride (nranks) live on the top-level
   * dev_handle; the plugin's createContext writer and this
   * device-side reader MUST agree on this signalId-major convention
   * (they are compiled in separate repos — a layout disagreement is
   * a silent wrong-tuple read).
   *
   * NULL when nSignals == 0.
   *
   * Layout is shared with the plugin definition in aws-ofi-nccl
   * (nccl_ofi_gin_gdaki_dev.h) — keep them in sync. */
  uint16_t *sig_address_handles; /* [nSignals * nranks] */
  uint16_t *sig_remote_qpns;     /* [nSignals * nranks] */
  uint32_t *sig_qkey;            /* [nSignals * nranks] */

  /* Counter-target addressing for this (poster) endpoint.
   *
   * Three GPU-resident [nranks] arrays that let THIS endpoint post a
   * counter-only RDMA write (a counter was requested and no signal)
   * addressed to peer P's DATA endpoint. The data endpoint binds only
   * a FI_WRITE counter, NOT a FI_REMOTE_WRITE counter, so a write
   * landing there ticks this (sender-local) counter's FI_WRITE
   * completion WITHOUT firing any signal on the receiver.
   *
   * This is the mirror of sig_*: a counter-only Put must NOT use the
   * poster's plain per-peer table (address_handles), because on a
   * counter (sc) endpoint that resolves to peer's same-index sc
   * endpoint, whose FI_REMOTE_WRITE counter IS GIN signal C on the
   * receiver — spuriously incrementing it. Routing to peer's data EP
   * avoids that. The write still posts from this endpoint's QP (so the
   * correct local counter ticks); only the remote target differs.
   *
   * Indexed by peer only ([nranks], no signalId/counterId dimension):
   * every counter on this rank targets the same peer data EP. Resolved
   * through this endpoint's OWN AV against peer P's data EP address.
   * Built for the data endpoint and every sc endpoint.
   *
   * Layout is shared with the plugin definition in aws-ofi-nccl
   * (nccl_ofi_gin_gdaki_dev.h) — keep them in sync. */
  uint16_t *cnt_address_handles; /* [nranks] */
  uint16_t *cnt_remote_qpns;     /* [nranks] */
  uint32_t *cnt_qkey;            /* [nranks] */
};

/*
 * Per-signal/counter endpoint handle, returned to device code through
 * dev_handle->signal_handles[] and dev_handle->counter_handles[].
 *
 * Composes nccl_ofi_gin_gdaki_dev_endpoint_handle (qp / cq / addressing /
 * sq_lock / counter completion tracking) and adds the cntr_value
 * pointer that the kernel reads to observe signal arrivals
 * (FI_REMOTE_WRITE) or counter increments (FI_WRITE).
 */
struct nccl_ofi_gin_gdaki_dev_counter_handle {
  /* Endpoint-common fields (qp, cq, addressing, sq_lock,
   * counter completion tracking). */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle base;
  /* NIC writes the hardware counter value here (GPU memory). Read
   * via system-scope atomic load (hwCounterLoad / waitSignal). */
  uint64_t *cntr_value;

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
  /* Data endpoint. The plugin binds a FI_WRITE counter to this
   * endpoint and populates data.local_cntr_value at createContext
   * time so the kernel can poll for local completion. */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle data;

  /* Per-counter / per-signal endpoint handles, populated when the
   * caller asked createContext for nCounters / nSignals > 0. Both
   * arrays index into the same underlying signal/counter endpoint
   * (sc_endpoint) on the plugin side; they expose two views of that
   * endpoint with cntr_value pointing at the FI_WRITE counter
   * (counter_handles) or the FI_REMOTE_WRITE counter (signal_handles).
   * The array pointer is NULL when the corresponding count is zero. */
  struct nccl_ofi_gin_gdaki_dev_counter_handle **counter_handles; /* [nCounters] or NULL */
  struct nccl_ofi_gin_gdaki_dev_counter_handle **signal_handles;  /* [nSignals]  or NULL */
  int32_t nCounters;
  int32_t nSignals;

  int32_t nranks;
  int32_t rank;

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
  uint64_t *scratch_remote_addrs;
  uint32_t *scratch_remote_rkeys;
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
