#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// Wraps a CUDA runtime call; on failure prints file, line, and error string, then exits.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


// upsweep_kernel --
//
// One level of the up-sweep (reduce) phase. `twoD` is the stride for
// this level (2, 4, 8, ...). Each of the `numThreads` active threads
// owns one pair: the left child sits at i + twoD/2 - 1 and the parent
// (right child) at i + twoD - 1, where i = threadIndex * twoD.
__global__ void
upsweep_kernel(int* data, int twoD, int numThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numThreads) return;

    int i = idx * twoD;
    data[i + twoD - 1] += data[i + twoD / 2 - 1];
}


// downsweep_kernel --
//
// One level of the down-sweep phase. Same indexing as the up-sweep,
// but each thread does swap-and-add:
//   t = left; left = parent; parent = parent + t   (using the old parent)
__global__ void
downsweep_kernel(int* data, int twoD, int numThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numThreads) return;

    int i = idx * twoD;
    int left  = i + twoD / 2 - 1;
    int right = i + twoD - 1;

    int t = data[left];
    data[left] = data[right];
    data[right] += t;
}


// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
//
// This version is an in-place, work-efficient (up-sweep / down-sweep)
// scan on `result`. `input` is not used because `result` already holds
// a copy of the input values.
void exclusive_scan(int* input, int N, int* result)
{
    if (N <= 0) return;

    // Work on the power-of-two padded length.
    int rounded = nextPow2(N);

    // Zero the padding (elements N .. rounded-1) so the scan never adds
    // uninitialized memory. These values cannot affect results[0..N-1]
    // in an exclusive scan, but zeroing avoids signed-overflow on garbage.
    if (rounded > N) {
        CUDA_CHECK(cudaMemset(result + N, 0, (size_t)(rounded - N) * sizeof(int)));
    }

    // ---- Up-sweep: build partial sums up the tree ----
    for (int twoD = 2; twoD <= rounded; twoD *= 2) {
        int numThreads = rounded / twoD;
        int blocks = (numThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, twoD, numThreads);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Set the root (last element) to 0 ----
    CUDA_CHECK(cudaMemset(result + (rounded - 1), 0, sizeof(int)));

    // ---- Down-sweep: push prefix sums back down the tree ----
    for (int twoD = rounded; twoD >= 2; twoD /= 2) {
        int numThreads = rounded / twoD;
        int blocks = (numThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, twoD, numThreads);
        CUDA_CHECK(cudaGetLastError());
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    return 0; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}