#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel
torch::Tensor ssm_scan_cuda_fwd(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
);

// C++ Dispatcher
torch::Tensor ssm_scan_fwd(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
) {
    // Basic verification
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(dt.is_cuda(), "dt must be a CUDA tensor");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");

    return ssm_scan_cuda_fwd(x, dt, A, B, C, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ssm_scan_fwd", &ssm_scan_fwd, "Mamba SSM Scan Forward (CUDA)");
}
