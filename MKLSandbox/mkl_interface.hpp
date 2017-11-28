#ifndef MKL_INTERFACE_HPP_2311993517
#define MKL_INTERFACE_HPP_2311993517

#if defined(__cplusplus)

// forward declarations
struct DFTI_DESCRIPTOR;
// ~ forward declarations

namespace mkl {

enum MKL_LAYOUT     { RowMajor = 101, ColMajor = 102,                  };
enum MKL_TRANSPOSE  { NoTrans  = 111, Trans    = 112, ConjTrans = 113, };
enum MKL_UPLO       { Upper    = 121, Lower    = 122,                  };
enum MKL_DIAG       { NonUnit  = 131, Unit     = 131,                  };
enum MKL_SIDE       { Left     = 141, Right    = 142,                  };
enum MKL_STORAGE    { Packed   = 151,                                  };
enum MKL_IDENTIFIER { AMatrix  = 161, BMatrix  = 162,                  };

namespace detail {
  
  void* Allocate(int, int);
  void  Deallocate(int, int);

  void GeneralDoubleMatrixMultiplication(double const*, bool, bool, bool, bool, bool,
                                         double const*, bool, bool, bool, bool, bool,
                                         int, int, int, double*);
  void GeneralFloatMatrixMultiplication(float const*, bool, bool, bool, bool, bool,
                                        float const*, bool, bool, bool, bool, bool,
                                        int, int, int, float*);

  bool GeneralDoubleMatrixFastFourierTransform1D(double*, int, void*);
  bool GeneralDoubleMatrixFastFourierTransform2D(double*, int, int, void*);

  bool GeneralDoubleMatrixInverseFastFourierTransform1D(void*, int, void*);
  bool GeneralDoubleMatrixInverseFastFourierTransform2D(void*, int, void*);

} // ~ namespace detail

// ---------------------------- //
// ------- BLAS Level 1 ------- //
// ---------------------------- //

// Vector Scalar Product
void saxpy(int const, float  const,  float  const*, int, float *, int const);
void daxpy(int const, double const,  double const*, int, double*, int const);
void caxpy(int const, void   const*, void   const*, int, void  *, int const);
void zaxpy(int const, void   const*, void   const*, int, void  *, int const);

// Copy Vector
void scopy(int const, float  const*, int const, float *, int const);
void dcopy(int const, double const*, int const, double*, int const);
void ccopy(int const, void   const*, int const, void  *, int const);
void zcopy(int const, void   const*, int const, void  *, int const);

// Swap Vectors
void sswap(int, float *, int const, float *, int const);
void dswap(int, double*, int const, double*, int const);
void cswap(int, void  *, int const, void  *, int const);
void zswap(int, void  *, int const, void  *, int const);

// ---------------------------- //
// ------- BLAS Level 2 ------- //
// ---------------------------- //

// General Matrix - Vector Multiplication

void sgemv(MKL_LAYOUT, MKL_TRANSPOSE, 
           int const, int const, float const, 
           float const*, int const, 
           float const*, int const, 
           float const, float*, int const);

void dgemv(MKL_LAYOUT, MKL_TRANSPOSE,
           int const, int const,
           double const,
           double const*, int const,
           double const*, int const,
           double const, double*, int const);

void cgemv(MKL_LAYOUT, MKL_TRANSPOSE,
           int const, int const,
           void const*,
           void const*, int const,
           void const*, int const,
           void const*,
           void*, int const);

void zgemv(MKL_LAYOUT, MKL_TRANSPOSE,
           int const, int const,
           void const*,
           void const*, int const,
           void const*, int const,
           void const*,
           void*, int const);

// ---------------------------- //
// ------- BLAS Level 3 ------- //
// ---------------------------- //

// General Matrix - Matrix Multiplication

void sgemm(MKL_LAYOUT, MKL_TRANSPOSE, MKL_TRANSPOSE,
           int const, int const, int const,
           float const,
           float const*, int const,
           float const*, int const,
           float const,
           float*, int const);

void dgemm(MKL_LAYOUT, MKL_TRANSPOSE, MKL_TRANSPOSE,
  int const, int const, int const,
  double const,
  double const*, int const,
  double const*, int const,
  double const,
  double*, int const);

void cgemm(MKL_LAYOUT, MKL_TRANSPOSE, MKL_TRANSPOSE,
  int const, int const, int const,
  void const*,
  void const*, int const,
  void const*, int const,
  void const*,
  void*, int const);

void zgemm(MKL_LAYOUT, MKL_TRANSPOSE, MKL_TRANSPOSE,
  int const, int const, int const,
  void const*,
  void const*, int const,
  void const*, int const,
  void const*,
  void*, int const);

// Symmetric Matrix - Matrix Multiplication

void ssymm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           int const, int const,
           float const,
           float const*, int const,
           float const*, int const,
           float*, int const);

void dsymm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           int const, int const,
           double const,
           double const*, int const,
           double const*, int const,
           double*, int const);

void csymm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           int const, int const,
           void const*,
           void const*, int const,
           void const*, int const,
           void*, int const);

void zsymm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           int const, int const,
           void const*,
           void const*, int const,
           void const*, int const,
           void*, int const);

// Triangular Matrix - Matrix Multiplication

void strmm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           MKL_TRANSPOSE const, MKL_DIAG const, int const, int const,
           float const,
           float const*, int const,
           float const*, int const);

void dtrmm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           MKL_TRANSPOSE const, MKL_DIAG const, int const, int const,
           double const,
           double const*, int const,
           double const*, int const);

void ctrmm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           MKL_TRANSPOSE const, MKL_DIAG const, int const, int const,
           void const*,
           void const*, int const,
           void const*, int const);

void ztrmm(MKL_LAYOUT const, MKL_SIDE const, MKL_UPLO const,
           MKL_TRANSPOSE const, MKL_DIAG const, int const, int const,
           void const*,
           void const*, int const,
           void const*, int const);

// ---------------------------- //
// ---------- LAPACK ---------- //
// ---------------------------- //

// Solve System of Linear Equations

bool
sgesv(MKL_LAYOUT const, int const, int const,
      float const*, int const, int const*,
      float*, int const);

bool
dgesv(MKL_LAYOUT const, int const, int const,
      double const*, int const, int const*,
      double*, int const);

bool
cgesv(MKL_LAYOUT const, int const, int const,
      void const*, int const, int const*,
      void*, int const);

bool
zgesv(MKL_LAYOUT const, int const, int const,
      void const*, int const, int const*,
      void*, int const);

namespace dfti {

typedef DFTI_DESCRIPTOR* DESCRIPTOR_HANDLE;

enum CONFIG_VALUE {

  // COMMIT_STATUS
  MKL_DFTI_COMMITED           = 30,
  MKL_DFTI_UNCOMMITED         = 31,

  // FORWARD_DOMAIN
  MKL_DFTI_COMPLEX            = 32,
  MKL_DFTI_REAL               = 33,

  // PRECISION
  MKL_DFTI_SINGLE             = 35,
  MKL_DFTI_DOUBLE             = 36,

  // COMPLEX_STORAGE and DFTI_CONJUGATE_EVEN_STORAGE
  MKL_DFTI_COMPLEX_COMPLEX    = 39,
  MKL_DFTI_COMPLEX_REAL       = 40,

  // REAL STORAGE
  MKL_DFTI_REAL_COMPLEX       = 41,
  MKL_DFTI_REAL_REAL          = 42,

  // DFTI PLACEMENT
  MKL_DFTI_INPLACE            = 43,
  MKL_DFTI_NOT_INPLACE        = 44,

  // DFTI_ORDERING
  MKL_DFTI_ORDERED            = 48,
  MLK_DFTI_BACKWARD_SCRAMBLED = 49,
  
  // Allow/Avoid Certain Usages
  MKL_DFTI_ALLOW              = 51,
  MKL_DFTI_AVOID              = 52,
  MKL_DFTI_NONE               = 53,

  // DFTI_PACKED_FORMAT (for storing conjugate-even finite sequence)
  MKL_DFTI_CCS_FORMAT         = 54,
  MKL_DFTI_PACK_FORMAT        = 55,
  MKL_DFTI_PERM_FORMAT        = 56,
  MKL_DFTI_CCE_FORMAT         = 57,

};

enum CONFIG_PARAM {
  
  // Domain for forward transform. No default value
  MKL_DFTI_FORWARD_DOMAIN         = 0,

  // Dimensionality, or rank. No default value
  MKL_DFTI_DIMENSION              = 1,

  // Length(s) of transform. No default value
  MKL_DFTI_LENGTHS                = 2,

  // Floating point precision, No default value
  MKL_DFTI_PRECISION              = 3,

  // Scale factor for forward transform [1.0]
  MKL_DFTI_FORWARD_SCALE          = 4,

  // Scale factor for backward transform [1.0]
  MKL_DFTI_BACKWARD_SCALE         = 5,

  // Number of data sets to be transformed
  MKL_DFTI_NUMBER_OF_TRANSFORMS   = 7,

  // Storage of finite complex-valued sequences in complex domain
  MKL_DFTI_COMPLEX_STORAGE        = 8,

  // Storage of finite real-valued sequences in real domain
  MKL_DFTI_REAL_STORAGE           = 9,

  // Storage of finite complex-valued sequences in conjugate-even domain [MKL_DFTI_COMPLEX_REAL]
  MKL_DFTI_CONJUGATE_EVEN_STORAGE = 10,

  // Placement of result [MKL_DFTI_INPLACE]
  MLK_DFTI_PLACEMENT              = 11,

  // Generalized strides for input data layout [tight, row-major for C]
  MKL_DFTI_INPUT_STRIDES          = 12,

  // Generalized strides for output data layout [tight, row-major for C]
  MKL_DFTI_OUTPUT_STRIDES         = 13,

  // Distance between first input elements for multiple tranforms [0]
  MKL_DFTI_INPUT_DISTANCE         = 14,

  // Distance between output elements for multiple transforms [0]
  MKL_DFTI_OUTPUT_DISTANCE        = 15,

  // Use of workspace during computation [DFTI_ALLOW]
  MKL_DFTI_WORKSPACE              = 17,

  // Ordering of the result [DFTI_ORDERED]
  MKL_DFTI_ORDERING               = 18,

  // Possible transposition of result [DFTI_NONE]
  MKL_DFTI_TRANSPOSE              = 19,

  // Packing format for MKL_DFTI_COMPLEX_REAL storage of finite conjugate-even sequences [DFT_CCS_FORMAT]
  MKL_DFTI_PACKED_FORMAT          = 21,

  // Commit status of the descriptor - R/O parameter
  MKL_DFTI_COMMIT_STATUS          = 22,

  // Version string for this DFTI implementation - R/O parameter
  MKL_DFTI_VERSION                = 23,

  // Number of user threads that share the descriptor [1]
  MKL_DFTI_NUMBER_OF_USER_THREADS = 26,

  // Limit the number of threads used by this descriptor [0 = don't care]
  MKL_DFTI_THREAD_LIMIT           = 27,

  // Possible input data destruction [MKL_DFTI_AVOID = prevent input data]
  MKL_DFTI_DESTROY_INPUT          = 28,
};


long int createDescriptor(DESCRIPTOR_HANDLE*, CONFIG_VALUE, CONFIG_VALUE, long int, ...);
long int copyDescriptor(DESCRIPTOR_HANDLE, DESCRIPTOR_HANDLE*&);
long int commitDescriptor(DESCRIPTOR_HANDLE);
long int computeForward(DESCRIPTOR_HANDLE, void*, ...);
long int computeBackward(DESCRIPTOR_HANDLE, void*, ...);
long int setValue(DESCRIPTOR_HANDLE, CONFIG_PARAM, ...);
long int getValue(DESCRIPTOR_HANDLE, CONFIG_PARAM, ...);
long int freeDescriptor(DESCRIPTOR_HANDLE*&);
char*    errorMessage(long int);
long int errorClass(long int, long int);

} // ~ namespace dfti

} // ~ namespace mkl

#endif // ~ if defined(___cplusplus)


#endif