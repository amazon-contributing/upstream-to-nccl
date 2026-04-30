// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_H
#define EFA_CUDA_DP_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "../common/efa_io_defs.h"

#define EFA_CUDA_DP_VERSION_MAJOR 0
#define EFA_CUDA_DP_VERSION_MINOR 0
#define EFA_CUDA_DP_VERSION_SUBMINOR 1

#ifdef __cplusplus
extern "C" {
#endif

enum efa_cuda_wc_opcode {
	EFA_CUDA_WC_SEND,
	EFA_CUDA_WC_RDMA_WRITE,
	EFA_CUDA_WC_RDMA_READ,
/*
 * Set value of EFA_CUDA_WC_RECV so consumers can test if a completion is a
 * receive by testing (opcode & EFA_CUDA_WC_RECV).
 */
	EFA_CUDA_WC_RECV                  = 1 << 7,
	EFA_CUDA_WC_RECV_RDMA_WITH_IMM,
};

struct efa_cuda_cq {
	uint64_t comp_mask;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t cc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_wq {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t max_batch;
	uint32_t wqes_pending;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	uint32_t pc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_rq {
	struct efa_cuda_wq wq;
};

struct efa_cuda_sq {
	struct efa_cuda_wq wq;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
};

struct efa_cuda_qp {
	uint64_t comp_mask;
	struct efa_cuda_sq sq;
	struct efa_cuda_rq rq;
};

struct efa_cuda_cq_attrs {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *buffer;
	uint32_t num_entries;
	uint32_t entry_size;
};

struct efa_cuda_qp_attrs {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *sq_buffer;
	uint8_t *rq_buffer;
	uint32_t *sq_doorbell;
	uint32_t *rq_doorbell;
	uint32_t sq_num_entries;
	uint32_t sq_entry_size;
	uint32_t sq_max_batch;
	uint32_t rq_num_entries;
	uint32_t rq_entry_size;
	uint32_t reserved;
};

struct efa_cuda_cq *efa_cuda_create_cq(struct efa_cuda_cq_attrs *attrs, uint32_t inlen);
void efa_cuda_destroy_cq(struct efa_cuda_cq *d_cq);
struct efa_cuda_qp *efa_cuda_create_qp(struct efa_cuda_qp_attrs *attrs, uint32_t inlen);
void efa_cuda_destroy_qp(struct efa_cuda_qp *d_qp);
int efa_cuda_get_version(int *major, int *minor, int *subminor);

#ifdef __cplusplus
}
#endif

#endif
