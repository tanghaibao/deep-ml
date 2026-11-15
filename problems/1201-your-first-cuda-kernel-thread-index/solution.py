#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

std::vector<int> global_thread_indices(int n) {
    // 1. allocate device memory for n ints
    // 2. launch the kernel with enough threads to cover n
    // 3. copy the result back to the host and return it
    std::vector<int> res(n);
    int *device_out = nullptr;
    size_t bytes = res.size() * sizeof(int);

    cudaMalloc(reinterpret_cast<void**>(&device_out), bytes);

    constexpr int threads = 256;
    int blocks = 1 + (n - 1) / threads;
    index_kernel<<<blocks, threads>>>(device_out, n);

    cudaMemcpy(res.data(), device_out, bytes, cudaMemcpyDeviceToHost);
    return res;
}
