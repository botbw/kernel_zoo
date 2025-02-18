// ref: https://docs.nvidia.cn/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k4-with-f64-floating-point-type
#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h>

dim3 grid(1);
dim3 block(32);

const int M = 8;
const int N = 8;
const int K = 4;
using A_TYPE = double;
using B_TYPE = double;
using C_TYPE = double;
using D_TYPE = double;

template<typename A_TYPE,
         typename B_TYPE,
         typename C_TYPE,
         typename D_TYPE,
         int M,
         int N,
         int K
         >
__global__ void mma_test(A_TYPE *A,B_TYPE *B,C_TYPE *C,D_TYPE *D){
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=d"(D[threadIdx.x*2]), "=d"(D[threadIdx.x*2+1])
      : "d"(A[threadIdx.x]), "d"(B[threadIdx.x%4*8+threadIdx.x/4]), "d"(C[threadIdx.x*2]), "d"(C[threadIdx.x*2+1]));
} 


template<typename A_TYPE,
         typename B_TYPE,
         typename C_TYPE,
         typename D_TYPE,
         int M,
         int N,
         int K
         >
__global__ void reference(A_TYPE *A,B_TYPE *B,C_TYPE *C,D_TYPE *D){
    for(int idx=threadIdx.x;idx<M*N;idx+=blockDim.x){
        int row = idx / N;
        int col = idx % N;
        D_TYPE acc = 0;
        for(int i=0;i<K;i++){
            acc += A[row*K+i] * B[i*N+col];
        }
        D[idx] = C[idx] + acc;
    }
}

template<typename T>
void print(std::string des,T *arr,int row,int col){
    std::cout << "-----" << des << "-----" << '\n';
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            std::cout << std::setw(4) << arr[i*col+j] << ' ';
        }
        std::cout << '\n';
    }
}

template<typename T>
void fill_matrix(T *arr,int size){
    for(int i=0;i<size;i++){
        arr[i] = rand()%64;
    }
}

template<typename T>
bool validate(T *arr,T *arr_ref,int size){
    for(int i=0;i<size;i++){
        if(arr[i] != arr_ref[i]){
            std::printf("at %d expected %f but got %f\n",i,arr_ref[i],arr[i]);
            return 0;
        }
    }
    return 1;
}

int main(){
    srand(time(NULL));

    A_TYPE *A;
    B_TYPE *B;
    C_TYPE *C;
    D_TYPE *D,*D_ref;

    cudaMallocManaged(&A,M*K*sizeof(A_TYPE));
    cudaMallocManaged(&B,K*N*sizeof(B_TYPE));
    cudaMallocManaged(&C,M*N*sizeof(C_TYPE));
    cudaMallocManaged(&D,M*N*sizeof(D_TYPE));
    cudaMallocManaged(&D_ref,M*N*sizeof(D_TYPE));

    fill_matrix<A_TYPE>(A,M*K);
    fill_matrix<B_TYPE>(B,K*N);
    fill_matrix<C_TYPE>(C,M*N);

    print("A",A,M,K);
    print("B",B,K,N);
    print("C",C,M,N);

    mma_test<A_TYPE,B_TYPE,C_TYPE,D_TYPE,M,N,K><<<grid,block>>>(A,B,C,D);

    reference<A_TYPE,B_TYPE,C_TYPE,D_TYPE,M,N,K><<<grid,block>>>(A,B,C,D_ref);

    cudaDeviceSynchronize();

    print("D",D,M,N);
    print("D_ref",D_ref,M,N);

    if(validate(D,D_ref,M*N)){
        std::printf("PASS\n");
    }else{
        std::printf("FAIL\n");
    }
}