pub(super) const FWD_SRC: &str = r#"
extern "C" __global__ void sdp_attn_fwd_kernel(
    const float* q, const float* k, const float* v, const float* mask,
    float* out, float* aw,
    unsigned int seq_q, unsigned int seq_k,
    unsigned int d_k, unsigned int d_v,
    unsigned int is_causal, float scale, unsigned int total,
    unsigned int has_mask, unsigned int mask_ndim, unsigned int num_heads
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int b = idx / seq_q;
    unsigned int i = idx % seq_q;
    const float* q_bi = q + (size_t)(b * seq_q + i) * d_k;
    float* aw_bi = aw + (size_t)(b * seq_q + i) * seq_k;
    float* out_bi = out + (size_t)(b * seq_q + i) * d_v;
    const float* k_b = k + (size_t)b * seq_k * d_k;
    const float* v_b = v + (size_t)b * seq_k * d_v;
    // Contiguous key-padding mask base: 2-D [batch_mask, seq_k] folds heads.
    size_t mask_row = (mask_ndim == 2u) ? (size_t)(b / num_heads) * seq_k : 0;

    // Phase 1: scores[j] = scale * dot(Q[i,:], K[j,:]); masked/causal -> -inf.
    float mx = -INFINITY;
    for (unsigned int j = 0; j < seq_k; ++j) {
        if (is_causal && j > i) { aw_bi[j] = -INFINITY; continue; }
        if (has_mask && mask[mask_row + j] == 0.0f) { aw_bi[j] = -INFINITY; continue; }
        const float* k_j = k_b + (size_t)j * d_k;
        float dot = 0.0f;
        for (unsigned int d = 0; d < d_k; ++d) dot = fmaf(q_bi[d], k_j[d], dot);
        float s = dot * scale;
        aw_bi[j] = s;
        if (s > mx) mx = s;
    }
    // Phase 2: numerically stable softmax over the row (exp(-inf)=0).
    float sum = 0.0f;
    for (unsigned int j = 0; j < seq_k; ++j) {
        float e = expf(aw_bi[j] - mx);
        aw_bi[j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (unsigned int j = 0; j < seq_k; ++j) aw_bi[j] *= inv;
    // Phase 3: out[i,l] = sum_j attn[i,j] * V[j,l].
    for (unsigned int l = 0; l < d_v; ++l) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < seq_k; ++j)
            acc = fmaf(aw_bi[j], v_b[(size_t)j * d_v + l], acc);
        out_bi[l] = acc;
    }
}
"#;

pub(super) const BWD_DQ_SRC: &str = r#"
extern "C" __global__ void sdp_attn_bwd_dq_kernel(
    const float* go, const float* k, const float* v, const float* aw,
    float* d_scores, float* gq,
    unsigned int seq_q, unsigned int seq_k,
    unsigned int d_k, unsigned int d_v,
    unsigned int has_gq, float scale, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int b = idx / seq_q;
    unsigned int i = idx % seq_q;
    const float* go_bi = go + (size_t)(b * seq_q + i) * d_v;
    const float* aw_bi = aw + (size_t)(b * seq_q + i) * seq_k;
    float* ds_bi = d_scores + (size_t)(b * seq_q + i) * seq_k;
    const float* v_b = v + (size_t)b * seq_k * d_v;
    const float* k_b = k + (size_t)b * seq_k * d_k;

    // d_attn_row[j] = dot(dO[i,:], V[j,:]) -> stash in d_scores row.
    for (unsigned int j = 0; j < seq_k; ++j) {
        const float* v_j = v_b + (size_t)j * d_v;
        float dot = 0.0f;
        for (unsigned int l = 0; l < d_v; ++l) dot = fmaf(go_bi[l], v_j[l], dot);
        ds_bi[j] = dot;
    }
    // rs = dot(A[i,:], d_attn_row).
    float rs = 0.0f;
    for (unsigned int j = 0; j < seq_k; ++j) rs = fmaf(aw_bi[j], ds_bi[j], rs);
    // d_scores[i,j] = A[i,j] * (d_attn_row[j] - rs)  (softmax backward).
    for (unsigned int j = 0; j < seq_k; ++j) ds_bi[j] = aw_bi[j] * (ds_bi[j] - rs);
    // dQ[i,d] += scale * sum_j d_scores[i,j] * K[j,d].
    if (has_gq) {
        float* gq_bi = gq + (size_t)(b * seq_q + i) * d_k;
        for (unsigned int d = 0; d < d_k; ++d) {
            float acc = 0.0f;
            for (unsigned int j = 0; j < seq_k; ++j)
                acc = fmaf(ds_bi[j], k_b[(size_t)j * d_k + d], acc);
            gq_bi[d] += acc * scale;
        }
    }
}
"#;

pub(super) const BWD_DKV_SRC: &str = r#"
extern "C" __global__ void sdp_attn_bwd_dkv_kernel(
    const float* go, const float* q, const float* aw, const float* d_scores,
    float* gk, float* gv,
    unsigned int seq_q, unsigned int seq_k,
    unsigned int d_k, unsigned int d_v,
    unsigned int has_gk, unsigned int has_gv, float scale, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int b = idx / seq_k;
    unsigned int j = idx % seq_k;
    // dK[j,d] += scale * sum_i d_scores[i,j] * Q[i,d].
    if (has_gk) {
        float* gk_bj = gk + (size_t)(b * seq_k + j) * d_k;
        for (unsigned int d = 0; d < d_k; ++d) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < seq_q; ++i) {
                float ds = d_scores[(size_t)(b * seq_q + i) * seq_k + j];
                float qv = q[(size_t)(b * seq_q + i) * d_k + d];
                acc = fmaf(ds, qv, acc);
            }
            gk_bj[d] += acc * scale;
        }
    }
    // dV[j,l] += sum_i A[i,j] * dO[i,l].
    if (has_gv) {
        float* gv_bj = gv + (size_t)(b * seq_k + j) * d_v;
        for (unsigned int l = 0; l < d_v; ++l) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < seq_q; ++i) {
                float awv = aw[(size_t)(b * seq_q + i) * seq_k + j];
                float gov = go[(size_t)(b * seq_q + i) * d_v + l];
                acc = fmaf(awv, gov, acc);
            }
            gv_bj[l] += acc;
        }
    }
}
"#;
