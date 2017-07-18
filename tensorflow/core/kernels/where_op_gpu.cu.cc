/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

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
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

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
