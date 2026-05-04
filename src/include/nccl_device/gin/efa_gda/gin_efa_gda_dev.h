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

struct nccl_ofi_gin_gdaki_dev_handle {
  void *qp;                  /* GPU-resident, layout-compatible with efa_cuda_qp */
  void *cq;                  /* GPU-resident, layout-compatible with efa_cuda_cq */
  uint16_t *address_handles; /* [nranks] in GPU mem */
  uint16_t *remote_qpns;    /* [nranks] in GPU mem */
  uint32_t *qkey;            /* [nranks] in GPU mem */
  void **signal_handles;     /* [nSignals] or NULL */
  void **counter_handles;    /* [nCounters] or NULL */
  uint64_t pending_reqs;
  int32_t nranks;
  int32_t rank;
};

struct nccl_ofi_gin_gdaki_mr_handle {
  uint32_t lkey;
  int32_t nranks;
  uint32_t rkeys[];          /* [nranks] flexible array */
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_ */
