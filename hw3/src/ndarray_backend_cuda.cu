#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int32_t index = gid;
    int32_t i_a = offset;
    for(int i = shape.size - 1; i >=0; i --) {
      i_a += (index % shape.data[i]) * strides.data[i];
      index = index / shape.data[i];      
    }

    out[gid] = a[i_a]; 

}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t index = gid;
    int32_t i_out = offset;
    for(int i = shape.size - 1; i >=0; i --) {
      i_out += (index % shape.data[i]) * strides.data[i];
      index = index / shape.data[i];      
    }

    out[i_out] = a[gid];
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */

    CudaDims dim = CudaOneDim(out->size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);

}


__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t index = gid;
    int32_t i_out = offset;
    for(int i = shape.size - 1; i >=0; i --) {
      i_out += (index % shape.data[i]) * strides.data[i];
      index = index / shape.data[i];      
    }

    out[i_out] = val;
}



void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

    CudaDims dim = CudaOneDim(out->size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);


}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

#define _MAX_(a, b) (((a) > (b))? (a) : (b))

__global__ void EwiseMulKernel      (const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] *   b[gid];      }
__global__ void EwiseDivKernel      (const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] /   b[gid];      }
__global__ void EwiseEqKernel       (const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] ==  b[gid];      }
__global__ void EwiseGeKernel       (const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] >=  b[gid];      }
__global__ void EwiseMaximumKernel  (const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = _MAX_(a[gid], b[gid]);  }


__global__ void ScalarMulKernel     (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] *   val;              }
__global__ void ScalarDivKernel     (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] /   val;              }
__global__ void ScalarEqKernel      (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] ==  val;              }
__global__ void ScalarGeKernel      (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = a[gid] >=  val;              }
__global__ void ScalarMaximumKernel (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = _MAX_(a[gid], val);          }

__global__ void ScalarPowerKernel   (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = std::pow(a[gid], val);       }


void EwiseMul       (const CudaArray& a, const CudaArray& b, CudaArray* out) {CudaDims dim = CudaOneDim(out->size); EwiseMulKernel    <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);  }
void EwiseDiv       (const CudaArray& a, const CudaArray& b, CudaArray* out) {CudaDims dim = CudaOneDim(out->size); EwiseDivKernel    <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);  }
void EwiseEq        (const CudaArray& a, const CudaArray& b, CudaArray* out) {CudaDims dim = CudaOneDim(out->size); EwiseEqKernel     <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);  }
void EwiseGe        (const CudaArray& a, const CudaArray& b, CudaArray* out) {CudaDims dim = CudaOneDim(out->size); EwiseGeKernel     <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);  }
void EwiseMaximum   (const CudaArray& a, const CudaArray& b, CudaArray* out) {CudaDims dim = CudaOneDim(out->size); EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);  }

void ScalarMul      (const CudaArray& a, scalar_t val, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); ScalarMulKernel    <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  }
void ScalarDiv      (const CudaArray& a, scalar_t val, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); ScalarDivKernel    <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  }
void ScalarEq       (const CudaArray& a, scalar_t val, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); ScalarEqKernel     <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  }
void ScalarGe       (const CudaArray& a, scalar_t val, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); ScalarGeKernel     <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  }
void ScalarMaximum  (const CudaArray& a, scalar_t val, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  }
void ScalarPower    (const CudaArray& a, scalar_t val, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); ScalarPowerKernel  <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  }  


__global__ void EwiseLogKernel     (const scalar_t* a, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = std::log(a[gid]);              }
__global__ void EwiseExpKernel     (const scalar_t* a, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = std::exp(a[gid]);              }
__global__ void EwiseTanhKernel    (const scalar_t* a, scalar_t* out, size_t size) { size_t gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid < size) out[gid] = std::tanh(a[gid]);             }


void EwiseLog      (const CudaArray& a, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); EwiseLogKernel    <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);  }
void EwiseExp      (const CudaArray& a, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); EwiseExpKernel    <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);  }
void EwiseTanh     (const CudaArray& a, CudaArray* out)  {CudaDims dim = CudaOneDim(out->size); EwiseTanhKernel   <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);  }



////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define BLOCK_TILE  8 // 8 * 8 elems for a block
#define THREAD_TILE 2 // 2 * 2 elems for a thread, 4 * 4 threads in a block
#define _MIN_(a, b) (((a) < (b))? (a) : (b))

__global__ void mm(const scalar_t* a, const scalar_t* b , scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
    __shared__ scalar_t sA[BLOCK_TILE][BLOCK_TILE];
    __shared__ scalar_t sB[BLOCK_TILE][BLOCK_TILE];
    scalar_t c[THREAD_TILE][THREAD_TILE] = {0};
    // scalar_t a[THREAD_TILE];
    // scalar_t b[THREAD_TILE];

    int yblock = blockIdx.y;
    int xblock = blockIdx.x;

    int ythread = threadIdx.y;
    int xthread = threadIdx.x;

    int m = _MIN_(M, (yblock + 1) * BLOCK_TILE) - yblock * BLOCK_TILE;
    int p = _MIN_(P, (xblock + 1) * BLOCK_TILE) - xblock * BLOCK_TILE;


    // start index in out of this block
    int32_t C_y = yblock * BLOCK_TILE;
    int32_t C_x = xblock * BLOCK_TILE;
    
    int32_t A_y = yblock * BLOCK_TILE;
    int32_t B_x = xblock * BLOCK_TILE;

    // start index in sX of this thread
    int s_y = ythread * THREAD_TILE;
    int s_x = xthread * THREAD_TILE; 

    for(int32_t ko = 0; ko < N; ko += BLOCK_TILE) {
        int n = _MIN_(N, ko + BLOCK_TILE) - ko;
        int32_t A_x = ko;
        int32_t B_y = ko;

        __syncthreads();
        // cooperative fetching:
        // sA[0:m][0:n] = A[yblock * L +: m][k +: n]
        // sB[0:n][0:p] = B[k +: n][xblock * L +: p]

        // for threads in block:
        // sX[ythread * V +: V, xtherad * V +: V]
        for(int i = 0; i < THREAD_TILE; i ++) {
            for(int j = 0; j < THREAD_TILE; j ++) {
                if((s_y + i < m) && (s_x + j < n)) {
                    sA[s_y + i][s_x + j] = a[(A_y + s_y + i) * N + (A_x + s_x + j)];
                }
                if((s_y + i < n) && (s_x + j < p)) {
                    sB[s_y + i][s_x + j] = b[(B_y + s_y + i) * N + (B_x + s_x + j)];
                }               
            }
        }

        __syncthreads();

        for(int i = 0; i < THREAD_TILE; i ++) {
            for(int j = 0; j < THREAD_TILE; j ++) {
                if((s_y + i < m) && (s_x + j < p)) {
                    for(int k = 0; k < n; k ++) {
                        c[i][j] += sA[s_y + i][k] * sB[k][s_x + j]; 
                    }
                }
            }
        }
    }

    // write back c to out
    for(int i = 0; i < THREAD_TILE; i ++) {
        for(int j = 0; j < THREAD_TILE; j ++) {
            if((C_y + s_y + i < M) && (C_x + s_x + j < P)) {
                out[(C_y + s_y + i) * P + (C_x + s_x + j)] = c[i][j];
            }
        }
    }

}




void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

    CudaDims dim;
    dim.grid = dim3(
        (M + BLOCK_TILE - 1) / BLOCK_TILE,
        (P + BLOCK_TILE - 1) / BLOCK_TILE,
        1
    );

    dim.block = dim3(
        BLOCK_TILE / THREAD_TILE,
        BLOCK_TILE / THREAD_TILE,
        1
    );

    mm<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t start_id = gid * reduce_size;
    scalar_t reducer = a[start_id];
    for(int32_t i = 1; i < reduce_size; i ++) {
        reducer = _MAX_(reducer, a[start_id + i]);
    }

    out[gid] = reducer;
}



void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

    CudaDims dim = CudaOneDim(out->size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t start_id = gid * reduce_size;
    scalar_t reducer = a[start_id];
    for(int32_t i = 1; i < reduce_size; i ++) {
        reducer = reducer + a[start_id + i];
    }

    out[gid] = reducer;
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

    CudaDims dim = CudaOneDim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
