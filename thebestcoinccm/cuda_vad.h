#pragma once


inline bool CudaCheck(cudaError_t err, const char * FUNC, int LINE)
{
    if (err != cudaSuccess)
    {
        printf("Cuda error in func '%s' at line %i : %s\n", FUNC, LINE, cudaGetErrorString(err));
        fprintf(stderr, "Cuda error in func '%s' at line %i : %s\n", FUNC, LINE, cudaGetErrorString(err));
        return false;
    }
    return true;
}


class CudaAlloc
{
    public:
        template <typename T>
        CudaAlloc(T * & ptr, size_t size) : m_ptr(*(void **)&ptr)
        {
            m_ptr = NULL;
            m_err = cudaMalloc(&m_ptr, size);
        }

        template <typename T>
        CudaAlloc(T * & ptr, size_t size, const char * FUNC, int LINE) : m_ptr(*(void **)&ptr)
        {
            m_ptr = NULL;
            m_err = cudaMalloc(&m_ptr, size);
            CudaCheck(m_err, FUNC, LINE);
        }

        ~CudaAlloc() { free(); }

        void free()
        {
            void * ptr = m_ptr;
            m_ptr = NULL;
            if (ptr)
                cudaFree(ptr);
        }

        operator bool () const { return (m_err == cudaSuccess); }
        cudaError_t error_code() const { return m_err; }
        const char * error_msg() const { return cudaGetErrorString(m_err); }

    private:
        void * &    m_ptr;
        cudaError_t m_err;

};


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
# define _ALIGN(x) __align__(x)
#elif _MSC_VER
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#endif


