#include "unitarize_utilities.h"
#include <cstdio>
#include <cuda.h>


namespace hisq {

  template<class T>
    __device__ __host__ T getTrace(const LinkVariable<T> & a)
    {
      return a(0,0) + a(1,1) + a(2,2);
    }


  template<class T>
    __device__ __host__  T getDeterminant(const LinkVariable<T> & a){

      T result;
      result = a(0,0)*(a(1,1)*a(2,2) - a(2,1)*a(1,2))
        - a(0,1)*(a(1,0)*a(2,2) - a(1,2)*a(2,0))
        + a(0,2)*(a(1,0)*a(2,1) - a(1,1)*a(2,0));

      return result;
    }


  template<class T>
    __device__ __host__ LinkVariable<T> operator+(const LinkVariable<T> & a, const LinkVariable<T> & b)
    {
      LinkVariable<T> result;
      for(int i=0; i<9; i++){
        result.data[i] = a.data[i] + b.data[i];
      }
      return result;
    }


  template<class T> 
    __device__ __host__ LinkVariable<T> operator-(const LinkVariable<T> & a, const LinkVariable<T> & b)
    {
      LinkVariable<T> result;
      for(int i=0; i<9; ++i){
        result.data[i] = a.data[i] - b.data[i];
      }
      return result;
    }


  template<class T, class S>
    __device__ __host__ LinkVariable<T> operator*(const S & scalar, const LinkVariable<T> & a){
      LinkVariable<T> result;
      for(int i=0; i<9; ++i){
        result.data[i] = scalar*a.data[i];
      }
      return result;
    }


  template<class T, class S>
    __device__ __host__ LinkVariable<T> operator*(const LinkVariable<T> & a, const S & scalar){
      return scalar*a;
    }


  template<class T>
    __device__ __host__
    LinkVariable<T> operator*(const LinkVariable<T> & a, const LinkVariable<T> & b)
    {
      // The compiler has a hard time unrolling nested loops,
      // so here I do it by hand. 
      // I could do something more sophisticated in the future.
      LinkVariable<T> result;
      result(0,0) = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0);
      result(0,1) = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1);
      result(0,2) = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2);
      result(1,0) = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0);
      result(1,1) = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1);
      result(1,2) = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2);
      result(2,0) = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0);
      result(2,1) = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1);
      result(2,2) = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2);
      return result;
    }

  template<class T>
    __device__ __host__
    LinkVariable<T> conj(const LinkVariable<T> & other){
      LinkVariable<T> result;
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          result(i,j) = conj(other(j,i));
        }
      }
      return result;
    }

  template<class T>
    __device__
    void loadLinkVariableFromArray(LinkVariable<T> *link, const T* const array, int dir, int idx, int stride)
    {
      for(int i=0; i<9; ++i){
        link->data[i] = array[idx + (dir*9 + i)*stride];
      }
      return;
    }

  template<class T>
    __device__
    void writeLinkVariableToArray(T* const array, const LinkVariable<T> & link,  int dir, int idx, int stride)
    {
      for(int i=0; i<9; ++i){ 
        array[idx + (dir*9 + i)*stride] = link.data[i];
      }
      return;
    }


  template<class Cmplx> // I need to change these to template - template parameters
    __device__  __host__
    void computeLinkInverse(LinkVariable<Cmplx>* uinv, const LinkVariable<Cmplx>& u)
    {

      const Cmplx & det = getDeterminant(u);
      const Cmplx & det_inv = getPreciseInverse(det);

      Cmplx temp;

      temp = u(1,1)*u(2,2) - u(1,2)*u(2,1);
      (*uinv)(0,0) = (det_inv*temp);

      temp = u(0,2)*u(2,1) - u(0,1)*u(2,2);
      (*uinv)(0,1) = (temp*det_inv);

      temp = u(0,1)*u(1,2)  - u(0,2)*u(1,1);
      (*uinv)(0,2) = (temp*det_inv);

      temp = u(1,2)*u(2,0) - u(1,0)*u(2,2);
      (*uinv)(1,0) = (det_inv*temp);

      temp = u(0,0)*u(2,2) - u(0,2)*u(2,0);
      (*uinv)(1,1) = (temp*det_inv);

      temp = u(0,2)*u(1,0) - u(0,0)*u(1,2);
      (*uinv)(1,2) = (temp*det_inv);

      temp = u(1,0)*u(2,1) - u(1,1)*u(2,0);
      (*uinv)(2,0) = (det_inv*temp);

      temp = u(0,1)*u(2,0) - u(0,0)*u(2,1);
      (*uinv)(2,1) = (temp*det_inv);

      temp = u(0,0)*u(1,1) - u(0,1)*u(1,0);
      (*uinv)(2,2) = (temp*det_inv);

      return;
    } 

} // end namespace hisq



