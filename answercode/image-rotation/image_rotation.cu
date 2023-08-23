#include <cstdio>

#include "image_rotation.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void rotate_image_kernel(float *input, float *output, int W, int H,
                        float sin_theta, float cos_theta, int num_src_images) {
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  int i = blockIdx.z;
  int dest_x = blockDim.x * blockIdx.x + threadIdx.x;
  int dest_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(dest_x > W || dest_y > H) return;

  // Rotate images
    float xOff = dest_x - x0;
    float yOff = dest_y - y0;
    int src_x = (int) (xOff * cos_theta + yOff * sin_theta + x0);
    int src_y = (int) (yOff * cos_theta - xOff * sin_theta + y0);
    if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
      output[i*H*W + dest_y * W + dest_x] = input[i*H*W + src_y * W + src_x];
      } else {
      output[i*H*W + dest_y * W + dest_x] = 0.0f;
    }
  }

// Device(GPU) pointers
static float *input_gpu, *output_gpu;

void rotate_image_naive(float *input_images, float *output_images, int W, int H,
                        float sin_theta, float cos_theta, int num_src_images) {
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  // Rotate images
  for (int i = 0; i < num_src_images; i++) {
    for (int dest_x = 0; dest_x < W; dest_x++) {
      for (int dest_y = 0; dest_y < H; dest_y++) {
        float xOff = dest_x - x0;
        float yOff = dest_y - y0;
        int src_x = (int) (xOff * cos_theta + yOff * sin_theta + x0);
        int src_y = (int) (yOff * cos_theta - xOff * sin_theta + y0);
        if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
          output_images[i * H * W + dest_y * W + dest_x] =
              input_images[i * H * W + src_y * W + src_x];
        } else {
          output_images[i * H * W + dest_y * W + dest_x] = 0.0f;
        }
      }
    }
  }
}

void rotate_image(float *input_images, float *output_images, int W, int H,
                  float sin_theta, float cos_theta, int num_src_images) {
  // (TODO) Upload input images to GPU
    CHECK_CUDA(cudaMemcpy(input_gpu, input_images, num_src_images*W*H*sizeof(float),cudaMemcpyHostToDevice));
    dim3 block_dim(32,32,1);
    dim3 grid_dim((W+block_dim.x-1)/block_dim.x,(H+block_dim.y-1)/block_dim.y,num_src_images);
    rotate_image_kernel<<<grid_dim,block_dim>>>(input_gpu,output_gpu,W,H,sin_theta,cos_theta,num_src_images);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_images,output_gpu,num_src_images*H*W*sizeof(float),cudaMemcpyDeviceToHost));

  // (TODO) Launch kernel on GPU
  // (TODO) Download output images from GPU

  // DO NOT REMOVE; NEED FOR TIME MEASURE
}

void rotate_image_init(int image_width, int image_height, int num_src_images) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&input_gpu,num_src_images*image_width*image_height*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&output_gpu,num_src_images*image_width*image_height*sizeof(float)));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_cleanup() {
  // (TODO) Free device memory
  CHECK_CUDA(cudaFree(input_gpu));
  CHECK_CUDA(cudaFree(output_gpu));
  

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
