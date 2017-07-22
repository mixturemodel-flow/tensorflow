<<<<<<< HEAD
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
=======
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
>>>>>>> 5620ccf... Implemented GPU-kernel for 'where'-op

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

<<<<<<< HEAD
=======
// See docs in ../ops/array_ops.cc.

>>>>>>> 5620ccf... Implemented GPU-kernel for 'where'-op
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

<<<<<<< HEAD
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "external/cub_archive/cub/device/device_reduce.cuh"
#include "external/cub_archive/cub/device/device_select.cuh"
#include "external/cub_archive/cub/iterator/counting_input_iterator.cuh"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/where_op.h"
=======
#include "tensorflow/core/kernels/where_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
>>>>>>> 5620ccf... Implemented GPU-kernel for 'where'-op
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

<<<<<<< HEAD
namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const typename Eigen::array<TIndex, NDIM> strides,
    int64* output) {
  // TODO(ebrevdo): Use a multi-dimensional loop, increasing the
  // dimensions of individual indices manually, instead of relying on
  // a scalar loop variable and using integer division.
  CUDA_1D_KERNEL_LOOP(i, output_rows) {
    TIndex index_value = ldg(output + NDIM * i);
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      *(output + NDIM * i + c) = index_value / strides[c];
      index_value %= strides[c];
    }
  }
}

template <typename TIndex>
struct NumTrue<GPUDevice, TIndex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const GPUDevice& d, TTypes<bool>::ConstFlat input,
      typename TTypes<TIndex>::Scalar num_true) {
    const cudaStream_t& cu_stream = GetCudaStream(ctx);

    std::size_t temp_storage_bytes = 0;
    const bool* input_data = input.data();
    TIndex* num_true_data = num_true.data();

    auto first_success =
        cub::DeviceReduce::Sum(/*temp_storage*/ nullptr, temp_storage_bytes,
                               /*d_in*/ input_data,
                               /*d_out*/ num_true_data,
                               /*num_items*/ input.size(),
                               /*stream*/ cu_stream);

    if (first_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceReduce::Sum to calculate "
          "temp_storage_bytes, status: ",
          cudaGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = cub::DeviceReduce::Sum(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_in*/ input_data,
        /*d_out*/ num_true_data,
        /*num_items*/ input.size(),
        /*stream*/ cu_stream);

    if (second_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceReduce::Sum to count "
          "number of true indices.  temp_storage_bytes: ",
          temp_storage_bytes, ", status: ", cudaGetErrorString(second_success));
    }

    return Status::OK();
  }
};

template struct NumTrue<GPUDevice, int32>;
template struct NumTrue<GPUDevice, int64>;

template <int NDIM>
class WhereOutputIterator {
 public:
  // Required iterator traits
  typedef WhereOutputIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef void value_type;
  typedef void pointer;
  typedef int64& reference;

#if (THRUST_VERSION >= 100700)
  // Use Thrust's iterator categories so we can use these iterators in Thrust
  // 1.7 (or newer) methods
  typedef typename thrust::detail::iterator_facade_category<
      thrust::device_system_tag, thrust::random_access_traversal_tag,
      value_type,
      reference>::type iterator_category;  ///< The iterator category
#else
  typedef std::random_access_iterator_tag
      iterator_category;  ///< The iterator category
#endif  // THRUST_VERSION

  WhereOutputIterator(int64* ptr, const Eigen::DenseIndex max_row)
      : ptr_(ptr), max_row_(max_row) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int64& operator[](int n) const {
    // If the selection mechanism finds too many true values (because
    // the input tensor changed between allocation of output and now),
    // we may accidentally try to write past the allowable memory.  If
    // valid is false, then we don't do this.  Instead, we'll read off
    // the number of items found in Flagged()'s d_num_selected_out at
    // the end and confirm that it matches the number of rows of output.
    const bool valid = FastBoundsCheck(n, max_row_);
    return *(ptr_ + (valid ? (NDIM * n) : 0));
  }

 private:
  int64* ptr_;
  const Eigen::DenseIndex max_row_;
};

template <typename TIndex, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<bool, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

template <int NDIM, typename Tindex>
struct Where<GPUDevice, NDIM, Tindex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const GPUDevice& d,
      typename TTypes<bool, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output, Tindex* found_true_host) {
    if (output.dimension(0) == 0) {
      // Nothing to do.
      return Status::OK();
    }

    const cudaStream_t& cu_stream = GetCudaStream(ctx);

    std::size_t temp_storage_bytes = 0;

    cub::CountingInputIterator<Tindex> select_counter(0);

    Tensor found_true_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<Tindex>::v(),
                                          TensorShape({}), &found_true_t));
    Tindex* found_true_device = found_true_t.scalar<Tindex>().data();

    WhereOutputIterator<NDIM> output_iterator(
        output.data(),
        /* max_row */ output.dimension(0));

    auto first_success =
        cub::DeviceSelect::Flagged(/*temp_storage*/ nullptr, temp_storage_bytes,
                                   /*d_in*/ select_counter,
                                   /*d_flags*/ input.data(),
                                   /*d_out*/ output_iterator,
                                   /*d_num_selected_out*/ found_true_device,
                                   /*num_items*/ input.size(),
                                   /*stream*/ cu_stream);
    if (first_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceSelect::Flagged to calculate "
          "temp_storage_bytes, status: ",
          cudaGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = cub::DeviceSelect::Flagged(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_in*/ select_counter,
        /*d_flags*/ input.data(),
        /*d_out*/ output_iterator,
        /*d_num_selected_out*/ found_true_device,
        /*num_items*/ input.size(),
        /*stream*/ cu_stream);

    if (second_success != cudaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch cub::DeviceSelect::Flagged to copy "
          "indices out, status: ",
          cudaGetErrorString(second_success));
    }

    // TODO(ebrevdo): Find a way to synchronously copy back data from
    // found_true_device to *found_true_host.

    const Eigen::array<Tindex, NDIM> strides =
        CalculateStrides<Tindex, NDIM>(input);
    const Tindex output_rows = output.dimension(0);
    CudaLaunchConfig config = GetCudaLaunchConfig(output_rows, d);
    PropagateWhereIndicesKernel<NDIM, Tindex>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            output_rows, strides, output.data());

    return Status::OK();
  }
};

#define DECLARE_GPU_SPEC_INDEX(Dims, Tindex) \
  template struct Where<GPUDevice, Dims, Tindex>
#define DECLARE_GPU_SPEC(Dims)         \
  DECLARE_GPU_SPEC_INDEX(Dims, int32); \
  DECLARE_GPU_SPEC_INDEX(Dims, int64)

DECLARE_GPU_SPEC(1);
DECLARE_GPU_SPEC(2);
DECLARE_GPU_SPEC(3);
DECLARE_GPU_SPEC(4);
DECLARE_GPU_SPEC(5);

#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_INDEX

}  // namespace functor

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
=======
#include "cuda/include/thrust/device_ptr.h"
#include "cuda/include/thrust/count.h"
#include "cuda/include/thrust/functional.h"
#include "cuda/include/thrust/copy.h"
#include "cuda/include/thrust/transform.h"
#include "cuda/include/thrust/iterator/counting_iterator.h"
#include "cuda/include/thrust/iterator/transform_iterator.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {
    
template <int NDIM, typename T>
struct __attribute__((__packed__)) fixed_array {
    T val[NDIM];
};

template <int NDIM>
struct WriteIndex : public thrust::unary_function<int64, fixed_array<NDIM, int64> > {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides;
  __device__ __host__ WriteIndex(const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides) : strides(strides) {}
  
  __device__ fixed_array<NDIM, int64> operator()(int64 index) const {
    fixed_array<NDIM, int64> r;
    for (int i = 0; i < NDIM; ++i) {
      r.val[i] = index / strides[i];
      index %= strides[i];
    }
    return r;
  }
};

template <int NDIM>
__global__ void CopyTransformIndices(
    int nthreads, Eigen::DSizes<Eigen::DenseIndex, NDIM> strides,
    const int64* indices, fixed_array<NDIM, int64>* output)
{
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    output[idx] = WriteIndex<NDIM>(strides)(indices[idx]);
  }
}

template <int NDIM>
struct ComputeWhere {
  
  static int64 Compute(OpKernelContext* ctx, const Tensor& input, const bool* input_flat,
    int64 num_true, int64* output)
  {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides;

    // Calculate strides for RowMajor order.
    EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input.tensor<bool, NDIM>())::Layout) ==
                         static_cast<int>(Eigen::RowMajor)),
                        INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);

    strides[NDIM - 1] = 1;
    for (int i = NDIM - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * input.dim_size(i + 1);
    }
    
    int64 num_elems = input.NumElements();
    thrust::device_ptr<const bool> input_begin(input_flat);
    
    //create tensor for the compacted indices
    TensorShape temp_shape({num_true});
    Tensor compacted_indices;
    if (!ctx->allocate_temp(DT_INT64, temp_shape, &compacted_indices).ok()) return -1;

    //fill linear indices
    thrust::device_ptr<int64> compacted_indices_ptr(compacted_indices.flat<int64>().data());
    int64 rsize = thrust::copy_if(
        thrust::counting_iterator<int64>(0), thrust::counting_iterator<int64>(num_elems),
        input_begin, compacted_indices_ptr, thrust::identity<bool>())
      - compacted_indices_ptr;
      
    //convert linear indices to coordinates
    //This throws an illegal memory access error
    //thrust::device_ptr<fixed_array<NDIM, int64> > output_begin((fixed_array<NDIM, int64>*) output);
    //thrust::transform(compacted_indices_ptr, compacted_indices_ptr + num_true, output_begin, WriteIndex<NDIM>(strides));
    
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    CudaLaunchConfig cfg = GetCudaLaunchConfig(num_true, d);
    CopyTransformIndices<NDIM> <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
        num_true, strides, compacted_indices.flat<int64>().data(), (fixed_array<NDIM, int64>*) output);
    
    //TODO: optimize this into one kernel call using copy_if and transform iterator
    return rsize;
  }
  
};

}

class WhereOpGPU : public OpKernel {
 public:
  explicit WhereOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();
    
    const bool* input_flat = input.flat<bool>().data();
    const thrust::device_ptr<const bool> input_begin(input_flat);
    const thrust::device_ptr<const bool> input_end(input_flat + input.NumElements());
    
    int64 num_true = thrust::count(input_begin, input_end, true);

    TensorShape output_shape({num_true, input_dims});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    if (num_true == 0) return; //fast exit

#define HANDLE_DIM(NDIM)                                             \
  case NDIM:                                                         \
    found_true = ComputeWhere<NDIM>::Compute(                        \
        context, input, input_flat,                                  \
        num_true, output->flat<int64>().data());                     \
    break;

    int64 found_true = 0;
    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "WhereOp : Unhandled input dimensions: ", input_dims));
    }
#undef HANDLE_DIM

    OP_REQUIRES(
        context, num_true == found_true,
        errors::InvalidArgument(
            "WhereOp: Race condition between counting the number of true "
            "elements and writing them.  When counting, saw ",
            num_true, " elements; but when writing their indices, saw ",
            found_true, " elements."));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhereOpGPU);
};

#define REGISTER_WHERE() \
  REGISTER_KERNEL_BUILDER(Name("Where").Device(DEVICE_GPU), WhereOpGPU);

REGISTER_WHERE();

}  // namespace tensorflow

#endif
>>>>>>> 5620ccf... Implemented GPU-kernel for 'where'-op
