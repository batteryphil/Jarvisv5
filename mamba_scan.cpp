#include <torch/extension.h>
#include <vector>

// Pure C++/LibTorch implementation of the Selective SSM Scan
// This replaces the slow Python loop with a compiled C++ version.

torch::Tensor ssm_scan_fwd(
    torch::Tensor x,      // (B, L, D)
    torch::Tensor dt,     // (B, L, D)
    torch::Tensor A,      // (D, N)
    torch::Tensor B_params, // (B, L, N)
    torch::Tensor C_params, // (B, L, N)
    torch::Tensor D       // (D)
) {
    const int B = x.size(0);
    const int L = x.size(1);
    const int D_dim = x.size(2);
    const int N = A.size(1);

    auto y = torch::zeros_like(x);
    auto h = torch::zeros({B, D_dim, N}, x.options());

    // Pre-calculate exp(A)
    // Note: In the selective version, A is usually fixed or slowly varying, 
    // but here we follow the Mamba discretization: A_bar = exp(dt * A)
    
    // We iterate through time L. 
    // This loop is now in C++, making it significantly faster than the Python version.
    for (int t = 0; t < L; ++t) {
        auto x_t = x.select(1, t);          // (B, D)
        auto dt_t = dt.select(1, t);        // (B, D)
        auto B_t = B_params.select(1, t);    // (B, N)
        auto C_t = C_params.select(1, t);    // (B, N)

        // Discretization: A_bar = exp(dt_t * A)
        // dt_t is (B, D), A is (D, N) -> dt_t.unsqueeze(-1) * A.unsqueeze(0) is (B, D, N)
        auto A_bar = torch::exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)); // (B, D, N)
        
        // B_bar = dt_t * B_t
        // dt_t is (B, D), B_t is (B, N) -> (B, D, 1) * (B, 1, N) = (B, D, N)
        auto B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1); // (B, D, N)

        // State update: h = A_bar * h + B_bar * x_t
        h = A_bar * h + B_bar * x_t.unsqueeze(-1); // (B, D, N)

        // Output: y = h @ C_t
        // h is (B, D, N), C_t is (B, N) -> (B, D, N) @ (B, N, 1) = (B, D, 1)
        y.select(1, t) = torch::matmul(h, C_t.unsqueeze(-1)).squeeze(-1);
    }

    return y + x * D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ssm_scan_fwd", &ssm_scan_fwd, "Mamba SSM Scan Forward (C++)");
}
