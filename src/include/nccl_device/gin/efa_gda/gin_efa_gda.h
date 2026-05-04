/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA implementations for NCCL GIN device-side APIs.
 *
 * This file provides ncclGinApi_*<NCCL_NET_DEVICE_GIN_EFA_GDA> template
 * specializations that target EFA via efa-dp-direct.
 *
 * Implemented: Put, Flush
 * Stub: PutValue, Get, FlushAsync, Wait, GetSignalPtr, GetCounterPtr,
 *       ResetSignal, ResetCounter (follow-up tasks)
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_H_

#include <cstdint>

#include "../gin_device_common.h"
#include "gin_efa_gda_dev.h"

/* efa-dp-direct device functions (inline implementations) */
#include "../../transport/net_efa_gda/efa-dp-direct/include/device/efa_cuda_dp_impl.cuh"

/* ── Helper: get device handle from GIN context ───────────────────── */

NCCL_DEVICE_INLINE static nccl_ofi_gin_gdaki_dev_handle*
efaGdaGetDevHandle(ncclGinCtx ctx) {
  return (nccl_ofi_gin_gdaki_dev_handle*)ctx.handle;
}

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
    coop.sync();
    if (coop.thread_rank() == 0 && hasWins && bytes > 0) {
      nccl_ofi_gin_gdaki_dev_handle *dev = efaGdaGetDevHandle(ctx);
      efa_cuda_qp *qp = (efa_cuda_qp *)dev->qp;
      nccl_ofi_gin_gdaki_mr_handle *dstMh = (nccl_ofi_gin_gdaki_mr_handle *)dstWin;
      nccl_ofi_gin_gdaki_mr_handle *srcMh = (nccl_ofi_gin_gdaki_mr_handle *)srcWin;

      efa_io_tx_wqe wr;
      efa_cuda_init_rdma_write_wr(&wr, 0, dstMh->rkeys[peer], dstOff);
      efa_cuda_wr_set_sge(&wr, srcMh->lkey, srcOff, (uint32_t)bytes);
      efa_cuda_wr_set_remote(&wr, dev->address_handles[peer],
                             (uint32_t)dev->remote_qpns[peer], dev->qkey[peer]);

      efa_cuda_start_sq_batch(qp, 1);
      efa_cuda_sq_batch_place_wr(qp, 0, &wr);
      efa_cuda_flush_sq_wrs(qp);

      atomicAdd((unsigned long long *)&dev->pending_reqs, 1ULL);
    }
    /* TODO: signal and counter handling (separate task) */
    (void)signal; (void)signalOp; (void)signalOpArg;
    (void)hasCounter; (void)counterId;
    (void)hasDescriptor; (void)descriptor;
    (void)required; (void)given; (void)optFlags;
    coop.sync();
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
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, uint32_t optFlags) {
    (void)ctx; (void)peer; (void)outRequest; (void)optFlags;
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
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)ord;
    coop.sync();
    if (coop.thread_rank() == 0) {
      nccl_ofi_gin_gdaki_dev_handle *dev = efaGdaGetDevHandle(ctx);
      efa_cuda_cq *cq = (efa_cuda_cq *)dev->cq;

      /* Poll CQ until all pending WRs are completed */
      uint64_t remaining = atomicAdd((unsigned long long *)&dev->pending_reqs, 0ULL);
      while (remaining > 0) {
        void *wc = efa_cuda_cq_poll(cq, 0);
        if (wc != nullptr) {
          efa_cuda_cq_pop(cq, 1);
          remaining = atomicAdd((unsigned long long *)&dev->pending_reqs, (unsigned long long)-1) - 1;
        }
        if (abortFlag && *abortFlag) break;
      }
    }
    coop.sync();
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    (void)ctx; (void)signalId;
    return nullptr;
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    (void)ctx; (void)counterId;
    return nullptr;
  }
};

/* ── ResetSignal ──────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    (void)ctx; (void)signal;
  }
};

/* ── ResetCounter ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    (void)ctx; (void)counterId;
  }
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_H_ */
