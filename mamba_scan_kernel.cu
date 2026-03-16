#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void ssm_scan_fwd_shared_kernel(
    const scalar_t* __restrict__ x,      // (B, L, D)
    const scalar_t* __restrict__ dt,     // (B, L, D)
    const scalar_t* __restrict__ A,      // (D, N)
    const scalar_t* __restrict__ B_in,   // (B, L, N)
    const scalar_t* __restrict__ C_in,   // (B, L, N)
    const scalar_t* __restrict__ D,      // (D)
    scalar_t* __restrict__ y,            // (B, L, D)
    int batch, int L, int D_dim, int N
) {
    // Grid: y-dim = batch, x-dim = d_model / blockDim.x
    int b = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= D_dim) return;

    // 🚀 ALLOCATE ULTRA-FAST L1 SHARED MEMORY
    // Shared across all threads in this block
    __shared__ float s_B[32]; 
    __shared__ float s_C[32];

    // Thread-local hardware registers for hidden state
    float h[32];
    #pragma unroll
    for (int n = 0; n < 32; ++n) h[n] = 0.0f;

    float d_val = (float)D[d];

    // 🚀 REGISTER CACHING: Load 'A' matrix once outside the loop
    float reg_A[32];
    for(int n=0; n<N; ++n){
        reg_A[n] = (float)A[d * N + n];
    }

    // The Main Sequence Loop
    for (int t = 0; t < L; ++t) {
        
        // 1. COLLABORATIVE LOAD: First N threads fetch B and C into Shared Memory
        if (threadIdx.x < N) {
            int bc_idx = b * L * N + t * N + threadIdx.x;
            s_B[threadIdx.x] = (float)B_in[bc_idx];
            s_C[threadIdx.x] = (float)C_in[bc_idx];
        }
        // Force all threads to wait until Shared Memory is loaded
        __syncthreads(); 

        int idx_bld = b * L * D_dim + t * D_dim + d;
        float x_t = (float)x[idx_bld];
        float dt_t = (float)dt[idx_bld];
        float y_t = 0.0f;

        // 2. THE MATH: Execute natively using cached registers and shared memory
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            float r = expf(dt_t * reg_A[n]);
            float b_bar = dt_t * s_B[n];

            h[n] = r * h[n] + b_bar * x_t;
            y_t += h[n] * s_C[n];
        }

        y[idx_bld] = (scalar_t)(y_t + x_t * d_val);
        
        // Force all threads to wait before the next loop overwrites Shared Memory
        __syncthreads(); 
    }
}

torch::Tensor ssm_scan_cuda_fwd(
    torch::Tensor x, torch::Tensor dt, torch::Tensor A,
    torch::Tensor B, torch::Tensor C, torch::Tensor D
) {
    const int batch = x.size(0);
    const int length = x.size(1);
    const int d_model = x.size(2);
    const int d_state = A.size(1);

    auto y = torch::empty_like(x);

    // Optimized grid alignment for Shared Memory blocks
    dim3 threads(256); 
    dim3 blocks((d_model + threads.x - 1) / threads.x, batch);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "ssm_scan_fwd_cuda", ([&] {
        ssm_scan_fwd_shared_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            D.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch, length, d_model, d_state
        );
    }));

    return y;
}
