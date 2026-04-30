/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA stub implementations for all NCCL GIN device-side APIs.
 *
 * This file provides ncclGinApi_*<NCCL_NET_DEVICE_GIN_EFA_GDA> template
 * specializations that target EFA via efa-dp-direct, replacing the
 * DOCA/ConnectX implementations in gin_gdaki.h.
 *
 * Current status: STUB — all functions compile and are safe to call
 * but do not perform real I/O. Real implementations will be added in
 * subsequent tasks (Put/PutValue, Signal, etc.).
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_H_

#include <cstdint>

#include "../gin_device_common.h"

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
    (void)ctx; (void)peer; (void)hasWins; (void)dstWin; (void)dstOff;
    (void)srcWin; (void)srcOff; (void)bytes; (void)signal; (void)signalOp;
    (void)signalOpArg; (void)hasCounter; (void)counterId; (void)hasDescriptor;
    (void)descriptor; (void)required; (void)given; (void)optFlags;
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
    coop.sync();
    (void)ctx; (void)hasDescriptor; (void)descriptor; (void)ord; (void)abortFlag;
    coop.sync();
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    (void)ctx; (void)signalId;
    return {nullptr, 0};
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    (void)ctx; (void)counterId;
    return {nullptr, 0};
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
