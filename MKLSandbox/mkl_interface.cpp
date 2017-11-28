#include <stdarg.h>

#include "mkl_service.h" // mkl_malloc, mkl_free
#include "mkl_cblas.h"   // BLAS Levels 1, 2 and 3
#include "mkl_lapacke.h" // LAPACK solvers
#include "mkl_dfti.h"    // fft

#include "mkl_interface.hpp"

namespace mkl {

// ---------------------------- //
// ------- BLAS Level 1 ------- //
// ---------------------------- //

// Vector - Scalar Product

//! \brief
//! - Computes a constant times a vector plus a vector (single-precision).
//! - After computation, the contents of vector y are replaced with the result. 
//! - The value computed is (alpha * x[i]) + y[i].
//!
//! \param[in] n    : Number of elements in the vectors..
//! \param[in] a    : Scaling factor for the values in x.
//! \param[in] x    : Input vector x.
//! \param[in] incx : Stride within x. For example, if incx is 7, every 7th element is used.
//! \param[in] y    : Input vector y.
//! \param[in] incy : Stride within y. For example, if incY is 7, every 7th element is used.
//!
//! \return void
void
saxpy(int const n, float const a, float const *x, int incx, float *y, int const incy) {
  cblas_saxpy(n, a, x, incx, y, incy);
}

//! \brief
//! - Computes a vector of double precision - double precision scalar products and adds the
//!   result to a vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] a    : Specifies the scalar a.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void
void
daxpy(int const n, double const a, double const *x, int incx, double *y, int const incy) {
  cblas_daxpy(n, a, x, incx, y, incy);
}

//! \brief
//! - Computes a vector of single precision complexes - single precision complex scalar 
//!   products and adds the result to a vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] a    : Specifies the scalar a.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void
void
caxpy(int const n, void const *a, void const *x, int incx, void *y, int const incy) {
  cblas_caxpy(n, a, x, incx, y, incy);
}

//! \brief
//! - Computes a vector of double precision complexes - double precision complex scalar 
//!   products and adds the result to a vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] a    : Specifies the scalar a.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void
void
zaxpy(int const n, void const *a, void const *x, int incx, void *y, int const incy) {
  cblas_zaxpy(n, a, x, incx, y, incy);
}

// Copy Vector

//! \brief
//! - Copies a single precision vector into another single precision vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
scopy(int n, float const *x, int const incx, float *y, int const incy) {
  cblas_scopy(n, x, incx, y, incy);
}

//! \brief
//! - Copies a double precision vector into another double precision vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
dcopy(int n, double const *x, int const incx, double *y, int const incy) {
  cblas_dcopy(n, x, incx, y, incy);
}

//! \brief
//! - Copies a single precision complex vector into another single complex precision
//!   vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
ccopy(int n, void const *x, int const incx, void *y, int const incy) {
  cblas_ccopy(n, x, incx, y, incy);
}

//! \brief
//! - Copies a double precision complex vector into another double complex precision
//!   vector.
//!
//! \param[in] n    : Specifies the number of elements in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incy)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
zcopy(int n, void const *x, int const incx, void *y, int const incy) {
  cblas_zcopy(n, x, incx, y, incy);
}

// Swap Vectors

//! \brief
//! - Exchanges the elements of two vectors (single precision).
//!
//! \param[in] n    : Number of elements in vectors.
//! \param[in] x    : Vector x. On return, contains elements copied from vector y.
//! \param[in] incx : Stride within x. For example, if incx is 7, every 7th element is used.
//! \param[in] y    : Vector y. On return, contains elements copied from vector x.
//! \param[in] incy : Stride within y. For example, if incy is 7, every 7th element is used.
//!
//! \return void.
void
sswap(int const n, float *x, int const incx, float *y, int const incy) {
  cblas_sswap(n, x, incx, y, incy);
}

//! \brief
//! - Swaps the elements of two double precision vectors.
//!
//! \param[in] n    : Specifies the number of element in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
dswap(int const n, double *x, int const incx, double *y, int const incy) {
  cblas_dswap(n, x, incx, y, incy);
}

//! \brief
//! - Swaps the elements of two single precision complex vectors.
//!
//! \param[in] n    : Specifies the number of element in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
cswap(int const n, void *x, int const incx, void *y, int const incy) {
  cblas_cswap(n, x, incx, y, incy);
}

//! \brief
//! - Swaps the elements of two double precision complex vectors.
//!
//! \param[in] n    : Specifies the number of element in vectors x and y.
//! \param[in] x    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incx : Specifies the increment for the elements of x.
//! \param[in] y    : Array, size at least (1 + (n - 1) * abs(incx)).
//! \param[in] incy : Specifies the increment for the elements of y.
//!
//! \return void.
void
zswap(int const n, void *x, int const incx, void *y, int const incy) {
  cblas_zswap(n, x, incx, y, incy);
}

// ---------------------------- //
// ------- BLAS Level 2 ------- //
// ---------------------------- //

// General Matrix - Vector Multiplication

//! \brief
//! - Computes a single precision general matrix - single precision vector product.
//!
//! - The sgemv routine performs a matrix - vector operation defined as:
//!   y := alpha*A*x + beta*y;
//!   or
//!   y := alpha*A'*x + beta*y;
//!   or
//!   y := alpha*conj(A')*x + beta*y;
//!
//!   where alpha and beta are scalars, x and y vectors and A is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     column-major (mkl::ColMajor).
//! \param[in] trans  : Specifies the operation:
//!                     - if trans=CblasNoTrans, then y := alpha*A*x + beta*y;
//!                     - if trans=CblasTrans, then y := alpha*A'*x + beta*y;
//!                     - if trans=CblasConjTrans then y := alpha*conj(A')*x + beta*y;
//! \param[in] m      : Specifies the number of rows of matrix A. The value of m must be at
//!                     least zero.
//! \param[in] n      : Specifies the number of columns of matrix A. The value of m must be
//!                     at least zero.
//! \param[in] alpha  : Specifies the alpha scalar.
//! \param[in] a      : Array, size lda*k.
//!                     For Layout=mkl::RowMajor, k is n. Before entry, the leading m-by-n
//!                     part of the array a must contain the matrix A. 
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program.
//!                     For Layout=mkl::RowMajor, the value of lda must be at least 
//!                     max(1, m). For Layout=mkl::ColMajor, the value of lda must be at 
//!                     least max(1, n).
//! \param[in] x      : Array size at least (1 + (n-1)*abs(incx) when trans=CblasNoTrans 
//!                     and at least (1 + (m-1)*abs(incx)) otherwise. Before entry, the
//!                     increment array x must contain the vector x.
//! \param[in] incx   : Specifies the increment for the elements of x. The value of incx
//!                     must not be zero.
//! \param[in] y      : Array, size at least (1 + (m-1)*abs(incy)) when trans=CblasNoTrans
//!                     and at least (1 + (n-1)*abs(incy)) otherwise. Before entry, the
//!                     increment array y must contain the vector y.
//! \param[in] incy   : Specifies the increment for the elements of y. The value of incy
//!                     must not be zero.
void
sgemv(MKL_LAYOUT const Layout, MKL_TRANSPOSE const trans,
      int const m, int const n,
      float const alpha,
      float const *a, int const lda,
      float const *x, int const incx,
      float const beta,
      float *y, int const incy) {
  cblas_sgemv((CBLAS_LAYOUT) Layout, (CBLAS_TRANSPOSE) trans, (MKL_INT) m, (MKL_INT) n,
              alpha, a, (MKL_INT) lda, x, (MKL_INT) incx, beta, y, (MKL_INT) incy);
}

//! \brief
//! - Computes a double precision general matrix - double precision vector product.
//!
//! - The dgemv routine performs a matrix - vector operation defined as:
//!   y := alpha*A*x + beta*y;
//!   or
//!   y := alpha*A'*x + beta*y;
//!   or
//!   y := alpha*conj(A')*x + beta*y;
//!
//!   where alpha and beta are scalars, x and y vectors and A is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     column-major (mkl::ColMajor).
//! \param[in] trans  : Specifies the operation:
//!                     - if trans=CblasNoTrans, then y := alpha*A*x + beta*y;
//!                     - if trans=CblasTrans, then y := alpha*A'*x + beta*y;
//!                     - if trans=CblasConjTrans then y := alpha*conj(A')*x + beta*y;
//! \param[in] m      : Specifies the number of rows of matrix A. The value of m must be at
//!                     least zero.
//! \param[in] n      : Specifies the number of columns of matrix A. The value of m must be
//!                     at least zero.
//! \param[in] alpha  : Specifies the alpha scalar.
//! \param[in] a      : Array, size lda*k.
//!                     For Layout=mkl::RowMajor, k is n. Before entry, the leading m-by-n
//!                     part of the array a must contain the matrix A. 
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program.
//!                     For Layout=mkl::RowMajor, the value of lda must be at least 
//!                     max(1, m). For Layout=mkl::ColMajor, the value of lda must be at 
//!                     least max(1, n).
//! \param[in] x      : Array size at least (1 + (n-1)*abs(incx) when trans=CblasNoTrans 
//!                     and at least (1 + (m-1)*abs(incx)) otherwise. Before entry, the
//!                     increment array x must contain the vector x.
//! \param[in] incx   : Specifies the increment for the elements of x. The value of incx
//!                     must not be zero.
//! \param[in] y      : Array, size at least (1 + (m-1)*abs(incy)) when trans=CblasNoTrans
//!                     and at least (1 + (n-1)*abs(incy)) otherwise. Before entry, the
//!                     increment array y must contain the vector y.
//! \param[in] incy   : Specifies the increment for the elements of y. The value of incy
//!                     must not be zero.
void
dgemv(MKL_LAYOUT const Layout, MKL_TRANSPOSE const trans,
  int const m, int const n,
  double const alpha,
  double const *a, int const lda,
  double const *x, int const incx,
  double const beta,
  double *y, int const incy) {
  cblas_dgemv((CBLAS_LAYOUT)Layout, (CBLAS_TRANSPOSE)trans, (MKL_INT)m, (MKL_INT) n,
    alpha, a, (MKL_INT)lda, x, (MKL_INT)incx, beta, y, (MKL_INT)incy);
}

//! \brief
//! - Computes a single precision complex general matrix - single precision complex
//!   vector product.
//!
//! - The cgemv routine performs a matrix - vector operation defined as:
//!   y := alpha*A*x + beta*y;
//!   or
//!   y := alpha*A'*x + beta*y;
//!   or
//!   y := alpha*conj(A')*x + beta*y;
//!
//!   where alpha and beta are scalars, x and y vectors and A is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     column-major (mkl::ColMajor).
//! \param[in] trans  : Specifies the operation:
//!                     - if trans=CblasNoTrans, then y := alpha*A*x + beta*y;
//!                     - if trans=CblasTrans, then y := alpha*A'*x + beta*y;
//!                     - if trans=CblasConjTrans then y := alpha*conj(A')*x + beta*y;
//! \param[in] m      : Specifies the number of rows of matrix A. The value of m must be at
//!                     least zero.
//! \param[in] n      : Specifies the number of columns of matrix A. The value of m must be
//!                     at least zero.
//! \param[in] alpha  : Specifies the alpha scalar.
//! \param[in] a      : Array, size lda*k.
//!                     For Layout=mkl::RowMajor, k is n. Before entry, the leading m-by-n
//!                     part of the array a must contain the matrix A. 
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program.
//!                     For Layout=mkl::RowMajor, the value of lda must be at least 
//!                     max(1, m). For Layout=mkl::ColMajor, the value of lda must be at 
//!                     least max(1, n).
//! \param[in] x      : Array size at least (1 + (n-1)*abs(incx) when trans=CblasNoTrans 
//!                     and at least (1 + (m-1)*abs(incx)) otherwise. Before entry, the
//!                     increment array x must contain the vector x.
//! \param[in] incx   : Specifies the increment for the elements of x. The value of incx
//!                     must not be zero.
//! \param[in] y      : Array, size at least (1 + (m-1)*abs(incy)) when trans=CblasNoTrans
//!                     and at least (1 + (n-1)*abs(incy)) otherwise. Before entry, the
//!                     increment array y must contain the vector y.
//! \param[in] incy   : Specifies the increment for the elements of y. The value of incy
//!                     must not be zero.
void
cgemv(MKL_LAYOUT const Layout, MKL_TRANSPOSE const trans,
  int const m, int const n,
  void const *alpha,
  void const *a, int const lda,
  void const *x, int const incx,
  void const *beta,
  float *y, int const incy) {
  cblas_cgemv((CBLAS_LAYOUT)Layout, (CBLAS_TRANSPOSE)trans, (MKL_INT)m, (MKL_INT) n,
    alpha, a, (MKL_INT)lda, x, (MKL_INT)incx, beta, y, (MKL_INT)incy);
}

//! \brief
//! - Computes a double precision complex general matrix - double precision complex
//!   vector product.
//!
//! - The zgemv routine performs a matrix - vector operation defined as:
//!   y := alpha*A*x + beta*y;
//!   or
//!   y := alpha*A'*x + beta*y;
//!   or
//!   y := alpha*conj(A')*x + beta*y;
//!
//!   where alpha and beta are scalars, x and y vectors and A is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     column-major (mkl::ColMajor).
//! \param[in] trans  : Specifies the operation:
//!                     - if trans=CblasNoTrans, then y := alpha*A*x + beta*y;
//!                     - if trans=CblasTrans, then y := alpha*A'*x + beta*y;
//!                     - if trans=CblasConjTrans then y := alpha*conj(A')*x + beta*y;
//! \param[in] m      : Specifies the number of rows of matrix A. The value of m must be at
//!                     least zero.
//! \param[in] n      : Specifies the number of columns of matrix A. The value of m must be
//!                     at least zero.
//! \param[in] alpha  : Specifies the alpha scalar.
//! \param[in] a      : Array, size lda*k.
//!                     For Layout=mkl::RowMajor, k is n. Before entry, the leading m-by-n
//!                     part of the array a must contain the matrix A. 
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program.
//!                     For Layout=mkl::RowMajor, the value of lda must be at least 
//!                     max(1, m). For Layout=mkl::ColMajor, the value of lda must be at 
//!                     least max(1, n).
//! \param[in] x      : Array size at least (1 + (n-1)*abs(incx) when trans=CblasNoTrans 
//!                     and at least (1 + (m-1)*abs(incx)) otherwise. Before entry, the
//!                     increment array x must contain the vector x.
//! \param[in] incx   : Specifies the increment for the elements of x. The value of incx
//!                     must not be zero.
//! \param[in] y      : Array, size at least (1 + (m-1)*abs(incy)) when trans=CblasNoTrans
//!                     and at least (1 + (n-1)*abs(incy)) otherwise. Before entry, the
//!                     increment array y must contain the vector y.
//! \param[in] incy   : Specifies the increment for the elements of y. The value of incy
//!                     must not be zero.
void
zgemv(MKL_LAYOUT const Layout, MKL_TRANSPOSE const trans,
  int const m, int const n,
  void const *alpha,
  void const *a, int const lda,
  void const *x, int const incx,
  void const *beta,
  float *y, int const incy) {
  cblas_zgemv((CBLAS_LAYOUT)Layout, (CBLAS_TRANSPOSE)trans, (MKL_INT)m, (MKL_INT)n,
    alpha, a, (MKL_INT)lda, x, (MKL_INT)incx, beta, y, (MKL_INT)incy);
}

// ---------------------------- //
// ------- BLAS Level 3 ------- //
// ---------------------------- //

// General Matrix - Matrix Multiplicatoin

//! \brief
//! - Computes single precision general matrix-matrix product.
//!
//! - The sgemm routine computes a scalar-matrix-matrix product and adds the result
//!   to a scalar-matrix product. The operation is defined as:
//!   C := alpha*op(A)*op(B) + beta*C;
//!   where:
//!   op(X) is one of op(X) = X, or op(X) = transpose(X), op(X) = hermitian_transpose(X),
//!   alpha and beta are scalars,
//!   A, B and C are matrices:
//!   op(A) is a m-by-k matrix,
//!   op(B) is a k-by-n matrix,
//!   C is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     colum-major (mkl::ColMajor).
//! \param[in] transa : Specifies the form of op(A) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(A)=A;
//!                     - if transa=mkl::Trans, then op(A)=transpose(A);
//!                     - if transa=mkl::ConjTrans, the op(A)=conjugate_transpose(A);
//! \param[in] transb : Specifies the form of op(B) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(B)=B;
//!                     - if transa=mkl::Trans, then op(B)=transpose(B);
//!                     - if transa=mkl::ConjTrans, the op(B)=conjugate_transpose(B);
//! \param[in] m      : Specifies the number of rows of the matrix op(A) and of the matrix 
//!                     C the value of m must be at least zero.
//! \param[in] n      : Specifies the number of columns of the matrix op(B) and of matrix
//!                     C the value of n must be at least zero.
//! \param[in] k      : Specifies the number of columns of matrix op(A) and the number
//!                     of rows of matrix op(B). The value of k must be at least zero.
//! \param[in] alpha  : Specifies the scalar alpha.
//! \param[in] a      : If array storage is row-major (mkl::RowMajor) and op(A)=A. Array
//!                     size is lda*m. Before entry, the leading k-by-m part of the array a
//!                     must contain the matrix A. If the array storage is row-major
//!                     (mkl::RowMajor) and op(A)=transpose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain the matrix A.
//!                     If array storage is column-major (mkl::ColMajor) and op(A) = A,
//!                     array size is lda*k. Before entry, the leading m-by-k part of the
//!                     array a must contain the matrix A. If array storage is column-major
//!                     (mkl::ColMajor) and op(A)=traspose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain matrix A.
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=A lda must be at least max(1,k).
//!                     If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and op(A)=A lda
//!                     must be at least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,k).
//! \param[in] b      : If array storage is row-major (mkl::RowMajor) and op(B)=B. Array
//!                     size is ldb by k. Before entry, the leading n-by-k part of the
//!                     array b must contain the matrix B. If array storage is row-major
//!                     (mkl::RowMajor) and op(B)=transpose(B) or op(B)=conjugate_trans(B),
//!                     array size ldb by n. Before entry, the leading k-by-n part of the
//!                     array a must contain the matrix B.
//!                     If array storage is column-major (mkl::ColMajor) and op(B)=B, array
//!                     size ldb by n. Before entry, the leading k-by-n part of the array b
//!                     must contain matrix B. If array storage is column major
//!                     (mkl::ColMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B), array size ldb by k. Before entry,
//!                     the leading n-by-k part of the array b must contain the matrix B.
//! \param[in] ldb    : Specifies the leading dimension of b as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(B)=B ldb must be at least max(1,n). If array storage is row-
//!                     major (mkl::RowMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B) ldb must be at least max(1,k).
//!                     If array storage is column-major (mkl::ColMajor) and op(B) = B
//!                     ldb must be at least max(1,k). If array storage is column-major
//!                     and op(B)=transpose(B) or op(B)=conjugate_transpose(B) ldb must be
//!                     at least max(1,n).
//! \param[in] beta   : Specifies the scalar beta.
//! \param[in] c      : If array storage is row-major (mkl::RowMajor) and op(B)=B, Array
//!                     size is ldc by m. Before entry, the leading n-by-m part of the
//!                     array c must contain the matrix C, expect when beta is equal to
//!                     zero, in which case c need not be set on entry.
//! \param[in] ldc    : Specifies the leading dimension of c as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) ldc
//!                     must be at least max(1,n). If array storage is colum-major
//!                     (mkl::ColMajor) ldc must be at least max(1,m).
//!
//! \return void.
void
sgemm(MKL_LAYOUT const Layout, MKL_TRANSPOSE const transa, MKL_TRANSPOSE const transb,
      int const m, int const n, int const k,
      float const alpha,
      float const *a, int const lda,
      float const *b, int const ldb,
      float const beta,
      float *c, int const ldc) {
  cblas_sgemm((CBLAS_LAYOUT) Layout, (CBLAS_TRANSPOSE) transa, (CBLAS_TRANSPOSE) transb,
              (MKL_INT) m, (MKL_INT) n, (MKL_INT) k, alpha,
              a, (MKL_INT) lda,
              b, (MKL_INT) ldb, beta,
              c, (MKL_INT) ldc);
}

//! \brief
//! - Computes double precision general matrix-matrix product.
//!
//! - The dgemm routine computes a scalar-matrix-matrix product and adds the result
//!   to a scalar-matrix product. The operation is defined as:
//!   C := alpha*op(A)*op(B) + beta*C;
//!   where:
//!   op(X) is one of op(X) = X, or op(X) = transpose(X), op(X) = hermitian_transpose(X),
//!   alpha and beta are scalars,
//!   A, B and C are matrices:
//!   op(A) is a m-by-k matrix,
//!   op(B) is a k-by-n matrix,
//!   C is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     colum-major (mkl::ColMajor).
//! \param[in] transa : Specifies the form of op(A) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(A)=A;
//!                     - if transa=mkl::Trans, then op(A)=transpose(A);
//!                     - if transa=mkl::ConjTrans, the op(A)=conjugate_transpose(A);
//! \param[in] transb : Specifies the form of op(B) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(B)=B;
//!                     - if transa=mkl::Trans, then op(B)=transpose(B);
//!                     - if transa=mkl::ConjTrans, the op(B)=conjugate_transpose(B);
//! \param[in] m      : Specifies the number of rows of the matrix op(A) and of the matrix 
//!                     C the value of m must be at least zero.
//! \param[in] n      : Specifies the number of columns of the matrix op(B) and of matrix
//!                     C the value of n must be at least zero.
//! \param[in] k      : Specifies the number of columns of matrix op(A) and the number
//!                     of rows of matrix op(B). The value of k must be at least zero.
//! \param[in] alpha  : Specifies the scalar alpha.
//! \param[in] a      : If array storage is row-major (mkl::RowMajor) and op(A)=A. Array
//!                     size is lda*m. Before entry, the leading k-by-m part of the array a
//!                     must contain the matrix A. If the array storage is row-major
//!                     (mkl::RowMajor) and op(A)=transpose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain the matrix A.
//!                     If array storage is column-major (mkl::ColMajor) and op(A) = A,
//!                     array size is lda*k. Before entry, the leading m-by-k part of the
//!                     array a must contain the matrix A. If array storage is column-major
//!                     (mkl::ColMajor) and op(A)=traspose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain matrix A.
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=A lda must be at least max(1,k).
//!                     If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and op(A)=A lda
//!                     must be at least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,k).
//! \param[in] b      : If array storage is row-major (mkl::RowMajor) and op(B)=B. Array
//!                     size is ldb by k. Before entry, the leading n-by-k part of the
//!                     array b must contain the matrix B. If array storage is row-major
//!                     (mkl::RowMajor) and op(B)=transpose(B) or op(B)=conjugate_trans(B),
//!                     array size ldb by n. Before entry, the leading k-by-n part of the
//!                     array a must contain the matrix B.
//!                     If array storage is column-major (mkl::ColMajor) and op(B)=B, array
//!                     size ldb by n. Before entry, the leading k-by-n part of the array b
//!                     must contain matrix B. If array storage is column major
//!                     (mkl::ColMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B), array size ldb by k. Before entry,
//!                     the leading n-by-k part of the array b must contain the matrix B.
//! \param[in] ldb    : Specifies the leading dimension of b as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(B)=B ldb must be at least max(1,n). If array storage is row-
//!                     major (mkl::RowMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B) ldb must be at least max(1,k).
//!                     If array storage is column-major (mkl::ColMajor) and op(B) = B
//!                     ldb must be at least max(1,k). If array storage is column-major
//!                     and op(B)=transpose(B) or op(B)=conjugate_transpose(B) ldb must be
//!                     at least max(1,n).
//! \param[in] beta   : Specifies the scalar beta.
//! \param[in] c      : If array storage is row-major (mkl::RowMajor) and op(B)=B, Array
//!                     size is ldc by m. Before entry, the leading n-by-m part of the
//!                     array c must contain the matrix C, expect when beta is equal to
//!                     zero, in which case c need not be set on entry.
//! \param[in] ldc    : Specifies the leading dimension of c as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) ldc
//!                     must be at least max(1,n). If array storage is colum-major
//!                     (mkl::ColMajor) ldc must be at least max(1,m).
//!
//! \return void.
void
dgemm(MKL_LAYOUT const Layout, MKL_TRANSPOSE const transa, MKL_TRANSPOSE const transb,
  int const m, int const n, int const k,
  double const alpha,
  double const *a, int const lda,
  double const *b, int const ldb,
  double const beta,
  double *c, int const ldc) {
  cblas_dgemm((CBLAS_LAYOUT)Layout, (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
    (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, alpha,
    a, (MKL_INT)lda,
    b, (MKL_INT)ldb, beta,
    c, (MKL_INT)ldc);
}

//! \brief
//! - Computes single precision complex general matrix-matrix product.
//!
//! - The cgemm routine computes a scalar-matrix-matrix product and adds the result
//!   to a scalar-matrix product. The operation is defined as:
//!   C := alpha*op(A)*op(B) + beta*C;
//!   where:
//!   op(X) is one of op(X) = X, or op(X) = transpose(X), op(X) = hermitian_transpose(X),
//!   alpha and beta are scalars,
//!   A, B and C are matrices:
//!   op(A) is a m-by-k matrix,
//!   op(B) is a k-by-n matrix,
//!   C is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     colum-major (mkl::ColMajor).
//! \param[in] transa : Specifies the form of op(A) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(A)=A;
//!                     - if transa=mkl::Trans, then op(A)=transpose(A);
//!                     - if transa=mkl::ConjTrans, the op(A)=conjugate_transpose(A);
//! \param[in] transb : Specifies the form of op(B) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(B)=B;
//!                     - if transa=mkl::Trans, then op(B)=transpose(B);
//!                     - if transa=mkl::ConjTrans, the op(B)=conjugate_transpose(B);
//! \param[in] m      : Specifies the number of rows of the matrix op(A) and of the matrix 
//!                     C the value of m must be at least zero.
//! \param[in] n      : Specifies the number of columns of the matrix op(B) and of matrix
//!                     C the value of n must be at least zero.
//! \param[in] k      : Specifies the number of columns of matrix op(A) and the number
//!                     of rows of matrix op(B). The value of k must be at least zero.
//! \param[in] alpha  : Specifies the scalar alpha.
//! \param[in] a      : If array storage is row-major (mkl::RowMajor) and op(A)=A. Array
//!                     size is lda*m. Before entry, the leading k-by-m part of the array a
//!                     must contain the matrix A. If the array storage is row-major
//!                     (mkl::RowMajor) and op(A)=transpose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain the matrix A.
//!                     If array storage is column-major (mkl::ColMajor) and op(A) = A,
//!                     array size is lda*k. Before entry, the leading m-by-k part of the
//!                     array a must contain the matrix A. If array storage is column-major
//!                     (mkl::ColMajor) and op(A)=traspose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain matrix A.
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=A lda must be at least max(1,k).
//!                     If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and op(A)=A lda
//!                     must be at least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,k).
//! \param[in] b      : If array storage is row-major (mkl::RowMajor) and op(B)=B. Array
//!                     size is ldb by k. Before entry, the leading n-by-k part of the
//!                     array b must contain the matrix B. If array storage is row-major
//!                     (mkl::RowMajor) and op(B)=transpose(B) or op(B)=conjugate_trans(B),
//!                     array size ldb by n. Before entry, the leading k-by-n part of the
//!                     array a must contain the matrix B.
//!                     If array storage is column-major (mkl::ColMajor) and op(B)=B, array
//!                     size ldb by n. Before entry, the leading k-by-n part of the array b
//!                     must contain matrix B. If array storage is column major
//!                     (mkl::ColMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B), array size ldb by k. Before entry,
//!                     the leading n-by-k part of the array b must contain the matrix B.
//! \param[in] ldb    : Specifies the leading dimension of b as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(B)=B ldb must be at least max(1,n). If array storage is row-
//!                     major (mkl::RowMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B) ldb must be at least max(1,k).
//!                     If array storage is column-major (mkl::ColMajor) and op(B) = B
//!                     ldb must be at least max(1,k). If array storage is column-major
//!                     and op(B)=transpose(B) or op(B)=conjugate_transpose(B) ldb must be
//!                     at least max(1,n).
//! \param[in] beta   : Specifies the scalar beta.
//! \param[in] c      : If array storage is row-major (mkl::RowMajor) and op(B)=B, Array
//!                     size is ldc by m. Before entry, the leading n-by-m part of the
//!                     array c must contain the matrix C, expect when beta is equal to
//!                     zero, in which case c need not be set on entry.
//! \param[in] ldc    : Specifies the leading dimension of c as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) ldc
//!                     must be at least max(1,n). If array storage is colum-major
//!                     (mkl::ColMajor) ldc must be at least max(1,m).
//!
//! \return void.
void
cgemm(MKL_LAYOUT const Layout, MKL_TRANSPOSE const transa, MKL_TRANSPOSE const transb,
  int const m, int const n, int const k,
  void const *alpha,
  void const *a, int const lda,
  void const *b, int const ldb,
  void const *beta,
  void *c, int const ldc) {
  cblas_cgemm((CBLAS_LAYOUT)Layout, (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
    (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, alpha,
    a, (MKL_INT)lda,
    b, (MKL_INT)ldb, beta,
    c, (MKL_INT)ldc);
}

//! \brief
//! - Computes double precision complex general matrix-matrix product.
//!
//! - The zgemm routine computes a scalar-matrix-matrix product and adds the result
//!   to a scalar-matrix product. The operation is defined as:
//!   C := alpha*op(A)*op(B) + beta*C;
//!   where:
//!   op(X) is one of op(X) = X, or op(X) = transpose(X), op(X) = hermitian_transpose(X),
//!   alpha and beta are scalars,
//!   A, B and C are matrices:
//!   op(A) is a m-by-k matrix,
//!   op(B) is a k-by-n matrix,
//!   C is a m-by-n matrix.
//!
//! \param[in] Layout : Specifies whether 2D array storage is row-major (mkl::RowMajor) or
//!                     colum-major (mkl::ColMajor).
//! \param[in] transa : Specifies the form of op(A) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(A)=A;
//!                     - if transa=mkl::Trans, then op(A)=transpose(A);
//!                     - if transa=mkl::ConjTrans, the op(A)=conjugate_transpose(A);
//! \param[in] transb : Specifies the form of op(B) used in the matrix multiplication:
//!                     - if transa=mkl::NoTrans, then op(B)=B;
//!                     - if transa=mkl::Trans, then op(B)=transpose(B);
//!                     - if transa=mkl::ConjTrans, the op(B)=conjugate_transpose(B);
//! \param[in] m      : Specifies the number of rows of the matrix op(A) and of the matrix 
//!                     C the value of m must be at least zero.
//! \param[in] n      : Specifies the number of columns of the matrix op(B) and of matrix
//!                     C the value of n must be at least zero.
//! \param[in] k      : Specifies the number of columns of matrix op(A) and the number
//!                     of rows of matrix op(B). The value of k must be at least zero.
//! \param[in] alpha  : Specifies the scalar alpha.
//! \param[in] a      : If array storage is row-major (mkl::RowMajor) and op(A)=A. Array
//!                     size is lda*m. Before entry, the leading k-by-m part of the array a
//!                     must contain the matrix A. If the array storage is row-major
//!                     (mkl::RowMajor) and op(A)=transpose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain the matrix A.
//!                     If array storage is column-major (mkl::ColMajor) and op(A) = A,
//!                     array size is lda*k. Before entry, the leading m-by-k part of the
//!                     array a must contain the matrix A. If array storage is column-major
//!                     (mkl::ColMajor) and op(A)=traspose(A) or
//!                     op(A)=conjugate_transpose(A), array size is lda*k. Before entry,
//!                     the leading m-by-k part of the array a must contain matrix A.
//! \param[in] lda    : Specifies the leading dimension of a as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=A lda must be at least max(1,k).
//!                     If array storage is row-major (mkl::RowMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and op(A)=A lda
//!                     must be at least max(1,m).
//!                     If array storage is column-major (mkl::ColMajor) and
//!                     op(A)=transpose(A) or op(A)=conjugate_transpose(A) lda must be at
//!                     least max(1,k).
//! \param[in] b      : If array storage is row-major (mkl::RowMajor) and op(B)=B. Array
//!                     size is ldb by k. Before entry, the leading n-by-k part of the
//!                     array b must contain the matrix B. If array storage is row-major
//!                     (mkl::RowMajor) and op(B)=transpose(B) or op(B)=conjugate_trans(B),
//!                     array size ldb by n. Before entry, the leading k-by-n part of the
//!                     array a must contain the matrix B.
//!                     If array storage is column-major (mkl::ColMajor) and op(B)=B, array
//!                     size ldb by n. Before entry, the leading k-by-n part of the array b
//!                     must contain matrix B. If array storage is column major
//!                     (mkl::ColMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B), array size ldb by k. Before entry,
//!                     the leading n-by-k part of the array b must contain the matrix B.
//! \param[in] ldb    : Specifies the leading dimension of b as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) and
//!                     op(B)=B ldb must be at least max(1,n). If array storage is row-
//!                     major (mkl::RowMajor) and op(B)=transpose(B) or
//!                     op(B)=conjugate_transpose(B) ldb must be at least max(1,k).
//!                     If array storage is column-major (mkl::ColMajor) and op(B) = B
//!                     ldb must be at least max(1,k). If array storage is column-major
//!                     and op(B)=transpose(B) or op(B)=conjugate_transpose(B) ldb must be
//!                     at least max(1,n).
//! \param[in] beta   : Specifies the scalar beta.
//! \param[in] c      : If array storage is row-major (mkl::RowMajor) and op(B)=B, Array
//!                     size is ldc by m. Before entry, the leading n-by-m part of the
//!                     array c must contain the matrix C, expect when beta is equal to
//!                     zero, in which case c need not be set on entry.
//! \param[in] ldc    : Specifies the leading dimension of c as declared in the calling
//!                     (sub)program. If array storage is row-major (mkl::RowMajor) ldc
//!                     must be at least max(1,n). If array storage is colum-major
//!                     (mkl::ColMajor) ldc must be at least max(1,m).
//!
//! \return void.
void
zgemm(MKL_LAYOUT const Layout, MKL_TRANSPOSE const transa, MKL_TRANSPOSE const transb,
  int const m, int const n, int const k,
  void const *alpha,
  void const *a, int const lda,
  void const *b, int const ldb,
  void const *beta,
  void *c, int const ldc) {
  cblas_zgemm((CBLAS_LAYOUT)Layout, (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb,
    (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, alpha,
    a, (MKL_INT)lda,
    b, (MKL_INT)ldb, beta,
    c, (MKL_INT)ldc);
}

// Symmetric Matrix - Matrix Multiplication

void
ssymm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
      int const m, int const n,
      float const alpha,
      float const *a, int const lda,
      float const *b, int const ldb,
      float const beta,
      float *c, int const ldc) {
  cblas_ssymm((CBLAS_LAYOUT) Layout, (CBLAS_SIDE) side, (CBLAS_UPLO) uplo,
              (MKL_INT) m, (MKL_INT) n, alpha, a, (MKL_INT) lda,
              b, (MKL_INT) ldb, beta, c, (MKL_INT) ldc);
}

void
dsymm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
  int const m, int const n,
  double const alpha,
  double const *a, int const lda,
  double const *b, int const ldb,
  double const beta,
  double *c, int const ldc) {
  cblas_dsymm((CBLAS_LAYOUT)Layout, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
    (MKL_INT)m, (MKL_INT)n, alpha, a, (MKL_INT)lda,
    b, (MKL_INT)ldb, beta, c, (MKL_INT)ldc);
}

void
csymm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
  int const m, int const n,
  void const *alpha,
  void const *a, int const lda,
  void const *b, int const ldb,
  void const *beta,
  void *c, int const ldc) {
  cblas_csymm((CBLAS_LAYOUT)Layout, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
    (MKL_INT)m, (MKL_INT)n, alpha, a, (MKL_INT)lda,
    b, (MKL_INT)ldb, beta, c, (MKL_INT)ldc);
}


void
zsymm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
      int const m, int const n,
      void const *alpha,
      void const *a, int const lda,
      void const *b, int const ldb,
      void const *beta,
      void *c, int const ldc) {
  cblas_csymm((CBLAS_LAYOUT)Layout, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
    (MKL_INT)m, (MKL_INT)n, alpha, a, (MKL_INT)lda,
    b, (MKL_INT)ldb, beta, c, (MKL_INT)ldc);
}

// Triangular Matrix-Matrix Multiplication

void
strmm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
      MKL_TRANSPOSE const transa, MKL_DIAG const diag, int const m, int const n,
      float const alpha,
      float const *a, int const lda,
      float *b, int const ldb) {
  cblas_strmm((CBLAS_LAYOUT) Layout, (CBLAS_SIDE) side, (CBLAS_UPLO) uplo,
              (CBLAS_TRANSPOSE) transa, (CBLAS_DIAG) diag, (MKL_INT) m, (MKL_INT) n, 
              alpha, a, (MKL_INT) lda, b, (MKL_INT) ldb);
}

void
dtrmm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
      MKL_TRANSPOSE const transa, MKL_DIAG const diag, int const m, int const n,
      double const alpha,
      double const *a, int const lda,
      double *b, int const ldb) {
  cblas_dtrmm((CBLAS_LAYOUT) Layout, (CBLAS_SIDE) side, (CBLAS_UPLO) uplo,
              (CBLAS_TRANSPOSE) transa, (CBLAS_DIAG) diag, (MKL_INT) m, (MKL_INT) n,
              alpha, a, (MKL_INT) lda, b, (MKL_INT) ldb);
}

void
ctrmm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
      MKL_TRANSPOSE const transa, MKL_DIAG const diag, int const m, int const n,
      void const *alpha,
      void const *a, int const lda,
      void *b, int const ldb) {
  cblas_ctrmm((CBLAS_LAYOUT)Layout, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag, (MKL_INT)m, (MKL_INT) n,
              alpha, a, (MKL_INT)lda, b, (MKL_INT)ldb);
}

void
ztrmm(MKL_LAYOUT const Layout, MKL_SIDE const side, MKL_UPLO const uplo,
      MKL_TRANSPOSE const transa, MKL_DIAG const diag, int const m, int const n,
      void const *alpha,
      void const *a, int const lda,
      void *b, int const ldb) {
  cblas_ztrmm((CBLAS_LAYOUT)Layout, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag, (MKL_INT)m, (MKL_INT) n,
              alpha, a, (MKL_INT)lda, b, (MKL_INT)ldb);
}

// ---------------------------- //
// ---------- LAPACK ---------- //
// ---------------------------- //

//! \brief
//! - Computes the solution to the system of linear equations with a square 
//!   coefficient single precision matrix A and multiple right-hand sides.
//!
//! - The routine solves for X the system of linear equations A*X=B; where A
//!   is a n-by-n matrix the columns of B are individual right-hand sides, and
//!   the columns of X are the corresponding solutions.
//!
//! \param[in] Layout : Specifies whether  matrix storage is row major.
//! \param[in] n      : The number of linear equations.
//! \param[in] nrhs   : The number of right-hand sides, the number of columns of
//!                     matrix E, cnrhs>0.
//! \param[in] a      : The array a(size max(1, lda*n)) contains the n-by-n
//!                     coefficient matrix A.
//! \param[in] lda    : The leading dimension of the array a; lda>=max(1,n).
//! \param[in] b      : The array b of size max(1, ldb*nrhs) for column major
//!                     layout and max(1, ldb*n) for row major layout contains
//!                     the n-by-nrhs matrix of right hand side matrix B.
//! \param[in] ldb    : The leading dimension of the array b; ldb>=max(1,n) for column
//!                     major layout and ldb>=nrhs for row major layout.
//!
//! \return true on successful execution, false othewise.
bool
sgesv(MKL_LAYOUT const Layout, int const n, int nrhs,
      float *a, int const lda, int *ipiv,
      float *b, int const ldb) {
  int Layout_ = (Layout == RowMajor)? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

  return !LAPACKE_sgesv(Layout_, n, nrhs, a, lda, ipiv, b, ldb);
}

//! \brief
//! - Computes the solution to the system of linear equations with a square 
//!   coefficient single precision matrix A and multiple right-hand sides.
//!
//! - The routine solves for X the system of linear equations A*X=B; where A
//!   is a n-by-n matrix the columns of B are individual right-hand sides, and
//!   the columns of X are the corresponding solutions.
//!
//! \param[in] Layout : Specifies whether  matrix storage is row major.
//! \param[in] n      : The number of linear equations.
//! \param[in] nrhs   : The number of right-hand sides, the number of columns of
//!                     matrix E, cnrhs>0.
//! \param[in] a      : The array a(size max(1, lda*n)) contains the n-by-n
//!                     coefficient matrix A.
//! \param[in] lda    : The leading dimension of the array a; lda>=max(1,n).
//! \param[in] b      : The array b of size max(1, ldb*nrhs) for column major
//!                     layout and max(1, ldb*n) for row major layout contains
//!                     the n-by-nrhs matrix of right hand side matrix B.
//! \param[in] ldb    : The leading dimension of the array b; ldb>=max(1,n) for column
//!                     major layout and ldb>=nrhs for row major layout.
//!
//! \return true on successful execution, false othewise.
bool
dgesv(MKL_LAYOUT const Layout, int const n, int nrhs,
      double *a, int const lda, int *ipiv,
      double *b, int const ldb) {
  int Layout_ = (Layout == RowMajor) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

  return !LAPACKE_dgesv(Layout_, n, nrhs, a, lda, ipiv, b, ldb);
}

//! \brief
//! - Computes the solution to the system of linear equations with a square 
//!   coefficient single precision complex matrix A and multiple right-hand sides.
//!
//! - The routine solves for X the system of linear equations A*X=B; where A
//!   is a n-by-n matrix the columns of B are individual right-hand sides, and
//!   the columns of X are the corresponding solutions.
//!
//! \param[in] Layout : Specifies whether  matrix storage is row major.
//! \param[in] n      : The number of linear equations.
//! \param[in] nrhs   : The number of right-hand sides, the number of columns of
//!                     matrix E, cnrhs>0.
//! \param[in] a      : The array a(size max(1, lda*n)) contains the n-by-n
//!                     coefficient matrix A.
//! \param[in] lda    : The leading dimension of the array a; lda>=max(1,n).
//! \param[in] b      : The array b of size max(1, ldb*nrhs) for column major
//!                     layout and max(1, ldb*n) for row major layout contains
//!                     the n-by-nrhs matrix of right hand side matrix B.
//! \param[in] ldb    : The leading dimension of the array b; ldb>=max(1,n) for column
//!                     major layout and ldb>=nrhs for row major layout.
//!
//! \return true on successful execution, false othewise.
bool
cgesv(MKL_LAYOUT const Layout, int const n, int nrhs,
      void *a, int const lda, int *ipiv,
      void *b, int const ldb) {
  int Layout_ = (Layout == RowMajor) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

  return !LAPACKE_cgesv(Layout_, n, nrhs, (MKL_Complex8*) a, lda, ipiv,
                        (MKL_Complex8*) b, ldb);
}

//! \brief
//! - Computes the solution to the system of linear equations with a square 
//!   coefficient double precision complex matrix A and multiple right-hand sides.
//!
//! - The routine solves for X the system of linear equations A*X=B; where A
//!   is a n-by-n matrix the columns of B are individual right-hand sides, and
//!   the columns of X are the corresponding solutions.
//!
//! \param[in] Layout : Specifies whether  matrix storage is row major.
//! \param[in] n      : The number of linear equations.
//! \param[in] nrhs   : The number of right-hand sides, the number of columns of
//!                     matrix E, cnrhs>0.
//! \param[in] a      : The array a(size max(1, lda*n)) contains the n-by-n
//!                     coefficient matrix A.
//! \param[in] lda    : The leading dimension of the array a; lda>=max(1,n).
//! \param[in] b      : The array b of size max(1, ldb*nrhs) for column major
//!                     layout and max(1, ldb*n) for row major layout contains
//!                     the n-by-nrhs matrix of right hand side matrix B.
//! \param[in] ldb    : The leading dimension of the array b; ldb>=max(1,n) for column
//!                     major layout and ldb>=nrhs for row major layout.
//!
//! \return true on successful execution, false othewise.
bool
zgesv(MKL_LAYOUT const Layout, int const n, int nrhs,
      void *a, int const lda, int *ipiv,
      void *b, int const ldb) {
  int Layout_ = (Layout == RowMajor) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

  return !LAPACKE_zgesv(Layout_, n, nrhs, (MKL_Complex16*) a, lda, ipiv, 
                        (MKL_Complex16*) b, ldb);
}

namespace dfti {

long int
createDescriptor(DESCRIPTOR_HANDLE *desc_handle, CONFIG_VALUE precision,
                 CONFIG_VALUE forward_domain, long int dimension, ...) {
  va_list args;

  va_start(args, dimension);

  MKL_LONG ret = DftiCreateDescriptor(desc_handle, (DFTI_CONFIG_VALUE) precision,
                                        (DFTI_CONFIG_VALUE) forward_domain, dimension,
                                        args);

  va_end(args);

  return ret;
}

long int
copyDescriptor(DESCRIPTOR_HANDLE desc_handle, DESCRIPTOR_HANDLE *&desc_handle_copy) {
  return DftiCopyDescriptor(desc_handle, desc_handle_copy);
}

long int
commitDescriptor(DESCRIPTOR_HANDLE desc_handle) {
  return DftiCommitDescriptor(desc_handle);
}

long int
computeForward(DESCRIPTOR_HANDLE desc_handle, void *x_inout, ...) {
  va_list args;

  va_start(args, x_inout);

  MKL_LONG ret = DftiComputeForward(desc_handle, x_inout, args);

  va_end(args);

  return ret;
}

long int
computeBackward(DESCRIPTOR_HANDLE desc_handle, void *x_inout, ...) {
  va_list args;

  va_start(args, x_inout);

  MKL_LONG ret = DftiComputeBackward(desc_handle, x_inout, args);

  va_end(args);

  return ret;
}

long int
setValue(DESCRIPTOR_HANDLE desc_handle, CONFIG_PARAM config_param, ...) {
  va_list args;

  va_start(args, config_param);

  MKL_LONG ret = DftiSetValue(desc_handle, (DFTI_CONFIG_PARAM) config_param, args);

  va_end(args);

  return ret;
}

long int
getValue(DESCRIPTOR_HANDLE desc_handle, CONFIG_PARAM config_param, ...) {
  va_list args;

  va_start(args, config_param);

  MKL_LONG ret = DftiGetValue(desc_handle, (DFTI_CONFIG_PARAM) config_param, args);

  va_end(args);

  return ret;
}

long int
freeDescriptor(DESCRIPTOR_HANDLE * const &desc_handle) {
  return DftiFreeDescriptor(desc_handle);
}

char*
errorMessage(long int status) {
  return DftiErrorMessage(status);
}

long int
errorClass(long int status, long int error_class) {
  return DftiErrorClass(status, error_class);
}

} // ~ namespace dfti

} // ~ namespace mkl

namespace gmath {

namespace detail {

// Allocator/Deallocator

void*
Allocate(int n_elem, int size) {
#if defined(GMATH_MKL_IS_AVAILABLE)
  return mkl_malloc(size * n_elem, GMATH_ALIGN_SIZE);
#elif defined(GMATH_POSIX_MEMALIGN_IS_AVAILABLE)
  void *out;
  if(posix_memalign((void**) &out,
                    (GMATH_ALIGN_SIZE >= sizeof(void*))? GMATH_ALIGN_SIZE : sizeof(void*),
                    size * n_elem));
#elif defined(GMATH_ALIGNED_MALLOC_IS_AVAILABLE)
  return _aligned_malloc(size * n_elem, GMATH_ALIGN_SIZE);
#else
  return malloc(size * n_elem);
#endif
}

void
Deallocate(void *mem) {
#if defined(GMATH_MKL_IS_AVAILABLE)
  mkl_free(mem);
#elif defined(GMATH_ALIGNED_MALLOC_IS_AVAILABLE)
  _aligned_free(mem);
#else
  free(mem);
#endif
}

// core operations

//! \brief
//! 



} // ~ namespace detail

} // ~ namespace gmath
