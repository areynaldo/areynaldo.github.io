---
title: "GPGPU"
date: 2026-01-27
summary: "Notes on GPU architecture: grids, blocks, warps, memory hierarchy, and CUDA execution model."
description: "Notes on GPU architecture: grids, blocks, warps, memory hierarchy, and CUDA execution model."
math: true
---

- **Grid**
    
    - Array of **Blocks** (1D, 2D or 3D)
    - Blocks of the same size **(up to 1024 Threads)**
    - $i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$
- **Block**
    
    - Array of **Threads** (1D, 2D or 3D)
- **Barrier Synchronization**
    
    - Threads wait for each other at same point
    - If waiting in different branches $\Rightarrow$ wait for each other forever (deadlock)
    - Not allowed between Blocks $\Rightarrow$ Blocks can execute in any order
- **Transparent Scalability**
    
    - Same code works on different hardware resources
    - Automatic scaling across different GPU architectures
- **Streaming Multiprocessor (SM)**
    
    - Device defines limit of Blocks and Threads per SM
    - e.g. Blackwell: 32 Blocks; 2048 Threads
        - $256 \text{ (Threads/block)} \cdot 8 \text{ (Blocks)} = 2048 \text{ (Threads)}$
        - $512 \text{ (Threads/block)} \cdot 4 \text{ (Blocks)} = 2048 \text{ (Threads)}$
- **Warp**
    
    - Scheduling purpose unit (fundamental execution unit)
        
    - A block assigned to a SM is divided into 32 thread scheduling units called **Warps**
        
    - Threads in warp executed following SIMD model
        
        - One instruction fetched and executed for all Threads in the warp
        - All Threads have the same execution timing (less control overhead)
        - Most efficient when same control-flow and branching
    - While a Warp waits for long instruction a ready Warp can be selected for execution (1 cycle context switch)
        
    - Latency tolerance / hiding (zero-overhead context switching)
        
    - Size of Warps are implementation-specific (currently 32 threads)
        
    - e.g.
        
        - Each Block has 256 Threads
        - Each Block has $256/32 = 8$ Warps
        - With 3 Blocks in each SM => $8 \cdot 3 = 24$ Warps in each SM
    - e.g. 8 Blocks and 1024 Threads per SM, 512 Threads per Block. Best thread block size?
        
        - 8x8 Blocks:
            - $8 \cdot 8 = 64$ Threads/Block
            - $1024/64 = 16$ Blocks/SM (but limited to 8 Blocks/SM)
            - $16 > 8 \Rightarrow$ limited by Block count
            - $64 \cdot 8 = 512$ (Threads/SM) $\Rightarrow$ resources underutilized because of fewer warps
        - 16x16 Blocks:
            - $16 \cdot 16 = 256$ Threads/Block
            - $1024/256 = 4$ Blocks/SM
            - $4 < 8 \Rightarrow$ only 50% of Block capacity used
            - $256 \cdot 4 = 1024$ Threads/SM $\Rightarrow$ Full Thread utilization
        - 32x32 Blocks:
            - $32 \cdot 32 = 1024$ Threads/Block
            - $1024 > 512$ Threads/Block limitation $\Rightarrow$ exceeds hardware limit
    - Linearized Thread Blocks are partitioned into Warps (Warp 0 starts with Thread 0)
        
    - Warp size can change with new architectures (currently 32 on NVIDIA GPUs)
        
    - Dependencies between threads solved with `__syncthreads()` - do not depend on Warp order
        
- **Control Divergence**
    
    - Threads in a warp take different branches (if-then-else / different loop iterations)
    - Branches are serialized (performance penalty)
        - Only threads in same current branch will be executed in parallel
        - Other threads wait idle
    - If branches on `threadIdx` $\Rightarrow$ divergence (threads within same warp differ)
    - If branches on `blockIdx` $\Rightarrow$ no divergence (all threads in warp take same path)
    - e.g.
        
        ```c
        __global__ void vecAddKernel(float* A, float* B, float* C, int n) {
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            if (i<n) {
                C[i] = A[i] + B[i];  // boundary check causes divergence}
            }
        }
        ```
        
    - Calculate how many warps are outside and how many inside the valid range
    - In divergent warps look for threads diverging
    - Differentiate between:
        - Type 1: Blocks that only diverge at the end (y-centered, x-outside at the end)
        - Type 2: Blocks that always diverge (y-outside-down)
        - e.g. Matrix Multiplication
            - 16x16 Tiles and Thread Blocks
            - $16 \times 16 / 32 = 256/32 = 8$ Warps
            - Matrix is 100x100
            - Each Thread has 7 Phases (ceiling of $100/16 = 6.25$)
            - 49 Thread Blocks ($7 \times 7$ grid)
            - Type 1:
                - $6 \cdot 7= 42$ Blocks with $8 \times 42 = 336$ Warps
                - $7 \cdot 336 = 2352$ Warp Phases
                - Only diverge in last phase
                - $1 \cdot 336 = 336$ Phases diverge
            - Type 2:
                - $7$ Blocks with $8 \times 7 = 56$ Warps
                - $7 \cdot 56 = 392$ Warp Phases
                - First 2 Warps in Block are within valid range
                - Other 6 Warps in Block are outside valid range
                - $2 \cdot 7 = 14$ Warp Phases diverge
            - Impact:
                - $(336+14)$ [Divergent Warp Phases] $/(2352+392)$ [Warp Phases] $= 0.1276$ (12.76% divergence)
- **Shared Memory**
    
    - Per SM (not per thread or per block - shared among blocks on same SM)
    - Way faster (latency and throughput) than global memory
    - Lifetime of thread block (released when block completes)
    - Load/store instructions (programmer-managed)
    - Temporary arena or cache-like usage
- **Tiling/Blocking**
    
    - Content from Global Memory divided into Tiles
    - Threads focus on few Tiles at a time
    - Good when Threads have similar memory access patterns
    - Core idea:
        - Identify Tile in Global Memory
        - Load Tile into Shared Memory cooperatively
        - Synchronize threads to start a Phase (e.g. Tile loaded)
        - Let Threads read from Shared Memory (fast access)
        - Let Threads compute using Shared Memory data
        - Synchronize threads to end a Phase (e.g. Tile processed)
        - Move to next Tile
    - Each Block should aim for max Threads
        - e.g. 16x16 Tile = 256 Threads
            - $2 \times 256$ loads
            - $256 \times (2 \times 16) = 8192$ mul/add ops
            - 16 flops/load
            - $16 \times 2 \times 256 \times 4\text{B} = 32768\text{B} = 32\text{KB}$ Shared Memory Required
                - If SM has 48KB Shared Memory $\Rightarrow$ 48KB/32KB = 1-2 Simultaneous Blocks in SM
        - e.g. 32x32 Tile = 1024 Threads
            - $2 \times 1024$ loads
            - $1024 \times (2 \times 32) = 65536$ mul/add ops
            - 32 flops/load
            - $32 \times 2 \times 1024 \times 4\text{B} = 262144\text{B} = 256\text{KB}$ Shared Memory Required
                - If SM with 48KB Shared Memory $\Rightarrow$ cannot fit (exceeds available memory)
    - Each `__syncthreads()` can reduce the count of active threads (acts as barrier)
        - $\Rightarrow$ More active threads can be more profitable for latency hiding
- **Matrix Multiplication Tile Size Handling**
    
    - If outside of valid range load 0 into `__shared__`
    - No impact on multiply + add (neutral element for addition)
    - Ensures correct results without boundary checks in computation loop
- **Padding**
    
    - To have a multiple of `TILE_SIZE` add elements
    - Big memory and transfer overhead (trade-off)
    - Simplifies indexing and removes boundary checks
- **Row Major**
    
    ```
    matrix = AAAA
             BBBB
             CCCC
             DDDD
    
    matrix_linearized = AAAABBBBCCCCDDDD
    ```
    
    - Default memory layout in C/CUDA
    - Adjacent columns in same row are adjacent in memory
- **Global Memory**
    
    - All Threads read from **Global Memory**
    - Memory access 4 bytes per floating-point addition $\Rightarrow$ 4B/FLOP bandwidth requirement
    - e.g. 1500 GFLOPS peak, 200GB/s DRAM bandwidth
        - $\frac{200[\text{GB/s}]}{4[\text{B/FLOP}]} = 50$ GFLOPS limit (memory-bound)
        - $50/1500 = 3.333%$ of max speed possible by the GPU
        - $\Rightarrow$ Reduce memory access count via reuse and shared memory
- **Textures**
    
    - Faster parallel access via graphic-specific caches (texture cache)
    - Read-only from kernel perspective
    - Good if locality (e.g. image processing, spatial locality)
    - Good if same data accessed by multiple threads (broadcast)
    - Hardware interpolation support
- **Surfaces**
    
    - Like Textures but writable
    - 1 mipmap level only
    - 2 APIs:
        - Texture Reference API (legacy, limited, everywhere)
        - Texture Object API (Compute Capability 3.0+, more flexible)
- **Memory Coalescing**
    
    - Mix multiple small memory accesses of parallel Threads into big memory access
    - If parallel Threads access the **same burst section only** in the same instruction (SIMD), only one DRAM access is needed
    - If multiple burst sections (no coalescing) $\Rightarrow$ more than one DRAM access needed
    - If not coalesced, unused data will be transferred with the burst section (wasted bandwidth)
    - Memory access in Warps are contiguous when `A[(expression independent of threadIdx.x) + threadIdx.x]`
    - Maximizes memory throughput efficiency
- **Corner Turning**
    
    - Row-wise access (coalesced) instead of Column-wise (strided)
    - Used for loading Tiles efficiently
    - Transpose data layout in shared memory if needed
- **DRAM Burst**
    

```
[0, 1, 2, 3][4, 5, 6, 7][8, 9, 10, 11][12, 13, 14, 15]
^           ^           ^              ^
Burst sections

16 Byte Memory Space
4 Byte Burst Sections
```

```
- Access to data transports entire burst section to GPU (not just requested byte)
- Minimum granularity of memory transfers
```

- **DRAM Core Organization**
    
    - DRAM Core Array around 16 Mbits
    - Stored in capacitor (needs refresh)
    
    ```
                    ┌─────────────┐      ┌────────────────┐
       Row   ──────►│    Row      │─────►│   Memory Cell  │
       Addr         │   Decoder   │      │   Core Array   │
                    └─────────────┘      └───────┬────────┘
                                                 ▼
                                         ┌────────────────┐
                                         │   Sense Amps   │
                                         └───────┬────────┘
                                                 ▼
                                         ┌────────────────┐
                                         │ Column Latches │
                                         └───────┬────────┘
                                                 │ Wide
                                                 ▼
                    Column  ────────────►┌──────────────┐
                    Addr                 │     Mux      │
                                         └──────┬───────┘
                                                │ Narrow
                                                ▼
                                          Pin Interface
                                         ───────────────
                                          Off-chip Data
    ```
    
    - Reading a cell from an Array is very slow (e.g. DDR3/GDDR4 Core Speed = 1/8 of Interface Speed)
    - SDRAM Cores cycle at 1/N of Interface Speed
        - Load N x Interface-Width DRAM Bits from the same row in an internal Buffer, then transport them in N steps
        - DDR3/DDR4: Buffer-Width = 8x Interface-Width
        - Like caching then sending each read Bit per Interface Speed
        - More Banks/Channels for better bandwidth (parallel access)
- **Intrinsics**
    
    - Functions that compile to single instructions (e.g., `__fmaf_rn()` for fused multiply-add)
    - Provide direct access to hardware operations
    - Typically faster than regular functions
- **Atomic Operations**
    
    - Hardware-atomic read-modify-write operation (>1000 cycles in DRAM global memory)
    - Other threads wait in queue (serialization)
    - Other atomic operations are sequential (one at a time)
    - Operations: `add, sub, inc, dec, min, max, exch, CAS (Compare and Swap)`
    - `atomicAdd(int *address, int value)`: read address, add value, store result at address and return old value
    - DRAM atomic operation starts with read: 100s of cycles
    - DRAM atomic operation ends with write: 100s of cycles
    - Memory access < 1/1000 of peak bandwidth if too many atomic ops (severe bottleneck)
    - Since Fermi: L2 Cache atomic ops latency average 1/10 of DRAM latency (shared between all Blocks)
    - Improvement via shared memory possible (much faster atomics)
- **Privatization**
    
    - Operations must be commutative and associative
    - Private data must fit in shared memory
    - Costs:
        - Overhead of preparing private copies (initialization)
        - Overhead of committing private copy to final copy (final atomic adds)
    - Benefits:
        - Less access control and serialization/sequential access
        - 10x+ performance improvements possible
        - Reduces global memory atomic contention
- **Naive Histogram: Partitions**
    
    - Split Input into Partitions per Thread
    - Add to bins sequentially
    - Contiguous threads do not access contiguous memory in parallel
    - No coalesced access $\Rightarrow$ Bad bandwidth usage
- **Naive Histogram: Interleaved**
    
    - Contiguous threads access contiguous memory in parallel
    - Coalesced access $\Rightarrow$ Good bandwidth usage
    - Still suffers from atomic contention
- **Reduction**
    
    - Input data order for processing not required (Associative and Commutative)
    - Divide into multiple parts
    - Let Threads process these parts in parallel
    - Reduction-tree to aggregate results (logarithmic depth)
    - Reduction operation (max, min, sum, mul)
    - User defined reduction operation possible:
        - Must be associative and commutative
        - Requires neutral element (identity element)
- **Efficient Sequential Reduction O(N)**
    
    - Initialize with neutral Element
    - Do reduction iterating input and updating result
    - Each input read only one time (efficient)
    - Single-threaded approach
- **Efficient Parallel Reduction O(log(N))**
    
    - Work: $(1/2)N + (1/4)N + (1/8)N + ... = N-1$
    - Average parallelism: $(N-1)/\log_2(N)$
        - Average: 50000 threads (example)
        - Peak: 500000 threads (example)
        - Not resource efficient (many threads idle after early steps)
    - Tournament-like structure (tree reduction)
    - Workload comparable to sequential (same total work)
    - Not all parallel algorithms are compute-efficient (some sacrifice efficiency for speed)
- **Parallel Sum Reduction**
    
    - 2 values per thread initially
    - Needed threads halved each iteration
    - $\log_2(n)$ steps and $n/2$ Threads needed initially
    - In-place implementation:
        - Input in global memory
        - Partial sum in shared memory
        - At start just copy from global memory
        - Block size limits n (shared memory size constraint)
    - Try to keep contiguous threads active to avoid divergence
        - Use threadIdx offsets (not modulo operations)
- **Direct Memory Access (DMA)**
    
    - Used with `cudaMemcpy()`
        - Performs 1 or more DMA transfers
        - Map address, check if page exists, check source and destination
        - For same DMA only 1 mapping necessary (cached)
    - Specialized HW data transport (no CPU involvement after setup)
    - Uses physical addresses or mapped IO spaces
    - PCIe interface (current standard)
    - OS could page out DMA received data but uses pinned memory instead (page-locked)
    - `cudaMemcpy()` is 2x faster if memory already pinned
- **Virtual Memory**
    
    - Maps virtual pointer addresses to physical addresses (via MMU hardware)
    - Paging mechanism (pages typically 4KB)
    - Allows isolation and flexible memory management
- **Device Overlapping**
    
    - Run Kernel while copying data (concurrent execution)
    - Divide into segments and do copying and compute in parallel for following segments
    - Pipeline approach for better throughput
- **Streams**
    
    - `cudaMemcpy()` and Kernel can execute in parallel (different streams)
    - Commands in FIFO-Queues (kernel and memcpy calls)
        - Driver and Device asynchronous
        - Driver makes sure commands run sequentially within a stream
    - Ops/Tasks in different streams can run in parallel (if hardware supports)
    - Host can check status with events (`cudaEventQuery()`)
- **Hyper Queues**
    
    - Multiple queues per engine (compute capability 3.5+)
    - More parallelism: some Streams of an Engine keep working, while others are blocked
    - Better resource utilization
- **Dynamic Parallelism**
    
    - Recursive Kernels (kernel launches kernel)
    - Compute capability >= 3.5 required
    - `-rdc=true` flag: multiple Device Code Objects linked (relocatable device code)
    - `cudadevrt.lib` makes part of Host API available for Device
    - Limits:
        - `cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, n);` - recursion depth limit
        - `cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, n);` - waiting kernel limit
    - No shared/local memory nor registers can be passed between parent and child kernels
    - Synchronization:
        - All threads in block wait at same point: `__syncthreads()`
        - Wait for child kernels: `cudaDeviceSynchronize()`
        - Usually: 1 Thread per block starts recursive call (avoid massive duplication)
- **Compute Graphics Interop**
    
    - Avoid GPU (Compute) -> CPU -> GPU (Graphics) data transfers
    - Direct sharing between compute and graphics pipelines
    - OpenGL:
        - Buffer Object (Vertex Buffer) maps to CUDA Buffer (1D or 2D array)
        - Textures map to CUDA Arrays
    - Steps:
        - Create Object (Graphics Context)
        - Register Object (Compute Context)
        - Map or Unmap Objects (exclusive access)
    - Textures:
        - Surface (write access from CUDA)
        - CUDA-Array (device arrays for copy operations, etc.)

#### Cheatsheet

```c
// Function qualifiers
__global__          // declares kernel function (called by host, runs on device)
__host__            // declares host only function (default)
__device__          // declares device only function
__host__ __device__ // same function for both host and device

// Variables                                 //  MEMORY      SCOPE   LIFETIME
                        int local_var;       //  register    thread  thread
__device__              int global_var;      //  global      grid    application
__device__ __shared__   int shared_var;      //  shared      block   block
__device__ __constant__ int constant_var;    //  constant    grid    application
// __device__ is optional for __shared__ and __constant__
// Automatic variables are stored in Registers
// Exceptions:
//  - per-Thread Arrays      -> global memory (spilled)
//  - no space in registers  -> global memory (register spilling)
//  - local memory is still slow (backed by global), same for stack (recursion)

// Malloc
cudaMalloc(&device_A, size);

// Free
cudaFree(device_A);

// Memcpy
cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(host_A, device_A, size, cudaMemcpyDeviceToHost);

// Host malloc (pinned memory for faster transfers)
cudaHostAlloc(ptr, size, cudaHostAllocDefault);

// Host free
cudaFreeHost(ptr);

// Error handling
cudaError_t err = cudaMalloc(&device_A, size);
if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}

// Kernel definition
__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Kernel launch
vecAddKernel<<<block_count, threads_per_block>>>(device_A, device_B, device_C, n);

// Grid and block dimensions
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);

dim3 dimGrid(ceil(n/256.0), 1, 1);
dim3 dimBlock(256, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);

vecAddKernel<<<ceil(n/256.0), 256>>>(...); // 1D only (simplified syntax)


// Synchronization
__syncthreads() // barrier synchronization (all threads in block wait for each other)
                // if put in different if statements they would wait for each other forever (deadlock)
```

#### Tiled Matrix Multiplication

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {

    // Define Tiles in shared memory
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    // Get Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate global Row and Col
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    // Accumulator for result
    float P_result_value = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {

        // Collaborative loading of M and N tiles into shared memory
        // Each thread loads one element from M and one from N
        ds_M[ty][tx] = M[Row*Width + ph*TILE_WIDTH+tx];
        ds_N[ty][tx] = N[(ph*TILE_WIDTH+ty)*Width + Col];

        __syncthreads(); // Wait for all threads to finish loading
        
        // Calculate using all data loaded in shared memory
        for (int i = 0; i < TILE_WIDTH; ++i) {
            P_result_value += ds_M[ty][i] * ds_N[i][tx];
        }
        __syncthreads(); // Wait before loading next tile
    }
    // Write result to global memory
    P[Row*Width+Col] = P_result_value;
}
```

#### Textures and Surfaces Cheatsheet

```c

struct cudaChannelFormatDesc {
    int x, y, z, w; // Bits per channel (e.g., 8, 16, 32)
    enum cudaChannelFormatKind f;
};

enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned;      // Signed integer
    cudaChannelFormatKindUnsigned;    // Unsigned integer
    cudaChannelFormatKindFloat;       // Floating point
};

struct cudaTextureDesc
{
    enum cudaTextureAddressMode addressMode[3]; // cudaAddressModeBorder: specified border color
                                                // cudaAddressModeClamp: value of border pixel
                                                // cudaAddressModeWrap: modulo, loop, repeat (only normalized coords)
                                                // cudaAddressModeMirror: mirrored (only normalized coords)
    enum cudaTextureFilterMode filterMode; // Interpolation if normalized-float access
                                           // cudaFilterModePoint: nearest neighbor
                                           // cudaFilterModeLinear: linear interpolation
    enum cudaTextureReadMode readMode; // cudaReadModeNormalizedFloat: [0.0..1.0] unsigned, [-1.0..1.0] signed
                                       // cudaReadModeElementType: raw values (pixel coords)
    int sRGB;                          // sRGB color space conversion
    int normalizedCoords;              // normalized coords [0..1]
                                       // alternative: pixel coords [0..width-1]
                                       // surfaces only support pixel coords
    unsigned int maxAnisotropy;
    enum cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
};

// Kernel example
__global__ void transformKernel(float* output, cudaTextureObject_t texObj, int width, int height, float theta) {
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float)width;
    float v = y / (float)height;
    
    // Transform coordinates (rotation)
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
    
    // Read from texture and write to global memory
    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

int main() {
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray; // Opaque type, only access via API functions, can change in future versions
    cudaMallocArray(&cuArray, &channelDesc, width, height); // Alloc and link array to format description

    // Copy to device memory some data located at address h_data in host memory
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);

    // Specify texture resource
    struct cudaResourceDesc resDesc; // Texture object description
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width, height, angle);

    // Destroy texture object
    cudaDestroyTextureObject(texObj);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);

    return 0;
}
```

#### Histogram (Naive with Atomics)

```c
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride is total number of threads (grid-stride loop)
    int stride = blockDim.x * gridDim.x;

    // All threads handle blockDim.x * gridDim.x consecutive elements (grid-stride pattern)
    while (i < size) {
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    }
}
```

#### Histogram with Privatization (Shared Memory)

```c
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo) {

    // Create private histogram in shared memory (bins)
    __shared__ unsigned int histo_private[7];

    // Initialize private histogram (first 7 threads do this)
    if (threadIdx.x < 7) histo_private[threadIdx.x] = 0;

    __syncthreads(); // Wait for initialization to complete
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride is total number of threads
    int stride = blockDim.x * gridDim.x;
    
    // Update private histogram (much faster atomics in shared memory)
    while (i < size) {
        atomicAdd(&(histo_private[buffer[i]/4]), 1);
        i += stride;
    }
    __syncthreads(); // Wait for all updates to complete

    // Commit private histogram to global final histogram (only once per block)
    if (threadIdx.x < 7) {
        atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    }
}
```

#### Naive Sum Reduction (Divergent)

```c
__shared__ float partialSum[2*BLOCK_SIZE];

unsigned int t = threadIdx.x;
unsigned int start = 2*blockIdx.x*blockDim.x;

// Load two elements per thread from global to shared memory
partialSum[t] = input[start + t];
partialSum[blockDim.x+t] = input[start + blockDim.x+t];

// Reduction in shared memory (divergent approach)
for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
{
    __syncthreads();
    if (t % stride == 0)  // Causes divergence (modulo operation)
        partialSum[2*t] += partialSum[2*t+stride];
}

// Host could add results itself or start another kernel

// Problems:
// - Half of threads in warps do nothing in every iteration (divergent warps)
// - Modulo operation causes branch divergence
// - Non-contiguous threads remain active
```

#### Better Sum Reduction (Minimized Divergence)

```c
// Same loading as above...

// Better reduction pattern (fewer divergent warps)
for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
    __syncthreads();
    if (t < stride) {  // Contiguous threads stay active
        partialSum[t] += partialSum[t+stride];
    }
}

// Benefits:
// - First warp fully active (no divergence within warp)
// - Contiguous threads reduce divergence
// - Better memory access patterns
```

#### Multi-Stream Example

```c
cudaStream_t stream0, stream1;

cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

float *d_A0, *d_B0, *d_C0; // Device memory for stream 0
float *d_A1, *d_B1, *d_C1; // Device memory for stream 1

// cudaMalloc() calls for d_A0, d_B0, d_C0, d_A1, d_B1, d_C1 go here

// Naive approach (limited overlap)
for (int i=0; i<n; i+=SegSize*2) {
    cudaMemcpyAsync(d_A0, h_A+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_B0, h_B+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
    cudaMemcpyAsync(h_C+i, d_C0, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
    
    cudaMemcpyAsync(d_A1, h_A+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
    vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
    cudaMemcpyAsync(h_C+i+SegSize, d_C1, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
}

// Better approach (maximal overlap - batch operations)
for (int i=0; i<n; i+=SegSize*2) {
    // Batch all H2D copies together
    cudaMemcpyAsync(d_A0, h_A+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_B0, h_B+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_A1, h_A+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);

    // Batch all kernel launches
    vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
    vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);

    // Batch all D2H copies together
    cudaMemcpyAsync(h_C+i, d_C0, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(h_C+i+SegSize, d_C1, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
}

cudaStreamSynchronize(stream_id); // Wait for all tasks in THAT STREAM to finish
cudaDeviceSynchronize();          // Wait for all tasks in ALL STREAMS to finish
```

#### Dynamic Parallelism Example

```c
__managed__ unsigned long long int facRes;

__global__ void
facCuda(const unsigned int n, unsigned long long int* res){

    if (n <= 1) {
        *res = 1; 
        return;
    }

    // Recursive kernel launch (only from device)
    facCuda<<<1, 1>>>(n - 1, res);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("%d!: %s\n", n, cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // Wait for child kernel to finish
    __syncthreads();         // Block-level synchronization

    *res = n * *res; // Multiply current n with factorial(n-1)
}

int main(void) {

    unsigned int n = 10;

    // Set recursion limits (important for dynamic parallelism)
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, n+1);
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

    facCuda<<<1, 1>>>(n, &facRes);
    cudaDeviceSynchronize(); // Wait for entire recursion tree

    printf("%d! = %lld\n", n, facRes);
}
```

#### Compute Graphics Interop Examples

```c

// ============================================
// Vertex Buffer Object (VBO) Interop
// ============================================
GLuint vboId;                   // OpenGL resource ID
cudaGraphicsResource_t vboRes;  // CUDA resource handle

// Reserve and create OpenGL buffer
glGenBuffers(1, &vboId);
glBindBuffer(GL_ARRAY_BUFFER, vboId);
glBufferData(GL_ARRAY_BUFFER, vboSize, 0, GL_DYNAMIC_DRAW); // Initialize
glBindBuffer(GL_ARRAY_BUFFER, 0);

// Register with CUDA for interop
cudaGraphicsGLRegisterBuffer(&vboRes, vboId, cudaGraphicsRegisterFlagsNone);

float* vboPtr;
while (!done) {
    // Reserve buffer for CUDA compute (lock from OpenGL)
    cudaGraphicsMapResources(1, &vboRes, 0); // count, resourceRef, stream

    // Get device pointer to mapped buffer
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, vboRes);
    runCUDA(vboPtr); // Compute kernel modifies vertex data

    // Return buffer to OpenGL (unlock)
    cudaGraphicsUnmapResources(1, &vboRes, 0);
    runGL(vboId); // Render using updated vertices
}

// ============================================
// Texture Interop
// ============================================
GLuint texId;
cudaGraphicsResource_t texRes;

// Reserve and create OpenGL texture
glGenTextures(1, &texId);
glBindTexture(GL_TEXTURE_2D, texId);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, texWidth, texHeight, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, 0);
glBindTexture(GL_TEXTURE_2D, 0);

// Register with CUDA
cudaGraphicsGLRegisterImage(&texRes, texId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

cudaArray* texArray;
<pixelType>* d_buffer; // Needs to get allocated as device memory

while (!done) {
    // Reserve texture for CUDA compute
    cudaGraphicsMapResources(1, &texRes, 0);

    // Get CUDA array pointer (texture data)
    cudaGraphicsSubResourceGetMappedArray(&texArray, texRes, 0, 0);
    
    runCUDA(d_buffer); // Compute kernel generates texture data
    
    // Copy computed data to texture array (device-to-device)
    cudaMemcpyToArray(texArray, 0, 0, d_buffer, d_buffer_size, cudaMemcpyDeviceToDevice);

    // Return texture to OpenGL
    cudaGraphicsUnmapResources(1, &texRes, 0);
    runGL(texId); // Render using updated texture
}
```