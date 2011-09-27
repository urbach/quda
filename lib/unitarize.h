#ifndef _UNITARIZE_H_
#define _UNITARIZE_H_



namespace hisq {
inline __device__ __host__ int index(int i, int j) { return i*3 + j; }

  // Given a real type T, returns the corresponding complex type
  template<class T>
  struct ComplexTypeId;

  template<>
  struct ComplexTypeId<float>
  {
    typedef float2 Type;
  };

  template<>
  struct ComplexTypeId<double>
  {
    typedef double2 Type;
  };

  template<class T> 
  struct RealTypeId; 
  
  template<>
  struct RealTypeId<float2>
  {
    typedef float Type;
  };

  template<>
  struct RealTypeId<double2>
  {
    typedef double Type;
  };


 __device__ __host__
 double2 makeComplex(const double & a, const double & b){
  return make_double2(a,b);
 }
  
  __device__ __host__
  float2 makeComplex(const float & a, const float & b){
    return make_float2(a,b);
  } 
  


  template<class Cmplx>
  __device__ __host__ Cmplx operator+(const Cmplx & a, const Cmplx & b){
    return makeComplex(a.x+b.x,a.y+b.y);
  }

  template<class Cmplx>
  __device__ __host__ Cmplx operator-(const Cmplx & a, const Cmplx & b)
  {
    return makeComplex(a.x-b.x,a.y-b.y);
  }

 template<class Cmplx>
  __device__ __host__ Cmplx operator*(const Cmplx & a, const typename RealTypeId<Cmplx>::Type & scalar)
  {
    return makeComplex(a.x*scalar,a.y*scalar);
  }

  template<class Cmplx>
  __device__ __host__ Cmplx operator*(const typename RealTypeId<Cmplx>::Type & scalar, const Cmplx & b)
  {
    return operator*(b,scalar);
  }

  template<class Cmplx>
  __device__ __host__ Cmplx operator*(const Cmplx & a, const Cmplx & b)
  {
    return makeComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
  }

  template<class Cmplx>
  __device__ __host__ Cmplx conj(const Cmplx & a)
  {
    return makeComplex(a.x,-a.y);
  }

  template<class Cmplx>
  __device__ __host__
  Cmplx getPreciseInverse(const Cmplx & z){
    typename RealTypeId<Cmplx>::Type ratio, max, denom;
    if( fabs(z.x) > fabs(z.y) ){ max = z.x; ratio = z.y/max; }else{ max=z.y; ratio = z.x/max; }
    denom = max*max*(1 + ratio*ratio);
    return makeComplex(z.x/denom, -z.y/denom);
  }


  // define the LinkVariable class
  template<class T> 
  class LinkVariable
  {
    public: 
	T data[9];

        // access matrix elements
        __device__ __host__ T const & operator()(int i, int j) const{
	  return data[index(i,j)];
        }

	// assign matrix elements
	__device__ __host__ T & operator()(int i, int j){
	  return data[index(i,j)];
        }
  };


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

} // end namespace hisq


#endif
