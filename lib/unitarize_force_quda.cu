#include <cstdio>
#include <quda_internal.h>
#include <unitarize_quda.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <read_gauge.h>
#include <gauge_quda.h>
#include <force_common.h>

#include "unitarize_utilities.h"

// constants - File scope only
__device__ __constant__ double FORCE_UNITARIZE_PI;
__device__ __constant__ double FORCE_UNITARIZE_PI23;
__device__ __constant__ double FORCE_UNITARIZE_EPS;
__device__ __constant__ double HISQ_FORCE_FILTER;


namespace hisq{
  namespace fermion_force{

    template<class Real>
     class DerivativeCoefficients{
       private:
         Real b[6]; 
                
         Real computeC00(const Real &, const Real &, const Real &);
         Real computeC01(const Real &, const Real &, const Real &);
         Real computeC02(const Real &, const Real &, const Real &);
         Real computeC11(const Real &, const Real &, const Real &);
         Real computeC12(const Real &, const Real &, const Real &);
         Real computeC22(const Real &, const Real &, const Real &);

       public:
         void set(const Real & u, const Real & v, const Real & w);
         Real getB00() const { return b[0]; }
         Real getB01() const { return b[1]; }
         Real getB02() const { return b[2]; }
         Real getB11() const { return b[3]; }
         Real getB12() const { return b[4]; }
         Real getB22() const { return b[5]; }
     };

    template<class Real>
      Real DerivativeCoefficients<Real>::computeC00(const Real & u, const Real & v, const Real & w){
        Real result =   -pow(w,3)*pow(u,6)
          + 3*v*pow(w,3)*pow(u,4)
          + 3*pow(v,4)*w*pow(u,4)
          -   pow(v,6)*pow(u,3)
          - 4*pow(w,4)*pow(u,3)
          - 12*pow(v,3)*pow(w,2)*pow(u,3)
          + 16*pow(v,2)*pow(w,3)*pow(u,2)
          + 3*pow(v,5)*w*pow(u,2)
          - 8*v*pow(w,4)*u
          - 3*pow(v,4)*pow(w,2)*u
          + pow(w,5)
          + pow(v,3)*pow(w,3);

        return result;
      }

    template<class Real>
      Real DerivativeCoefficients<Real>::computeC01(const Real & u, const Real & v, const Real & w){
        Real result =  - pow(w,2)*pow(u,7)
          - pow(v,2)*w*pow(u,6)
          + pow(v,4)*pow(u,5)   // This was corrected!
          + 6*v*pow(w,2)*pow(u,5)
          - 5*pow(w,3)*pow(u,4)    // This was corrected!
          - pow(v,3)*w*pow(u,4)
          - 2*pow(v,5)*pow(u,3)
          - 6*pow(v,2)*pow(w,2)*pow(u,3)
          + 10*v*pow(w,3)*pow(u,2)
          + 6*pow(v,4)*w*pow(u,2)
          - 3*pow(w,4)*u
          - 6*pow(v,3)*pow(w,2)*u
          + 2*pow(v,2)*pow(w,3);
        return result;
      }

    template<class Real>
      Real DerivativeCoefficients<Real>::computeC02(const Real & u, const Real & v, const Real & w){
        Real result =   pow(w,2)*pow(u,5)
          + pow(v,2)*w*pow(u,4)
          - pow(v,4)*pow(u,3)
          - 4*v*pow(w,2)*pow(u,3)
          + 4*pow(w,3)*pow(u,2)
          + 3*pow(v,3)*w*pow(u,2)
          - 3*pow(v,2)*pow(w,2)*u
          + v*pow(w,3);
        return result;
      }

    template<class Real>
      Real DerivativeCoefficients<Real>::computeC11(const Real & u, const Real & v, const Real & w){
        Real result = - w*pow(u,8)
          - pow(v,2)*pow(u,7)
          + 7*v*w*pow(u,6)
          + 4*pow(v,3)*pow(u,5)
          - 5*pow(w,2)*pow(u,5)
          - 16*pow(v,2)*w*pow(u,4)
          - 4*pow(v,4)*pow(u,3)
          + 16*v*pow(w,2)*pow(u,3)
          - 3*pow(w,3)*pow(u,2)
          + 12*pow(v,3)*w*pow(u,2)
          - 12*pow(v,2)*pow(w,2)*u
          + 3*v*pow(w,3);
        return result;
      }

    template<class Real>
      Real DerivativeCoefficients<Real>::computeC12(const Real & u, const Real & v, const Real & w){
        Real result =  w*pow(u,6)
          + pow(v,2)*pow(u,5) // Fixed this!
          - 5*v*w*pow(u,4)  // Fixed this!
          - 2*pow(v,3)*pow(u,3)
          + 4*pow(w,2)*pow(u,3)
          + 6*pow(v,2)*w*pow(u,2)
          - 6*v*pow(w,2)*u
          + pow(w,3);
        return result;
      }

    template<class Real>
      Real DerivativeCoefficients<Real>::computeC22(const Real & u, const Real & v, const Real & w){
        Real result = - w*pow(u,4)
          - pow(v,2)*pow(u,3)
          + 3*v*w*pow(u,2)
          - 3*pow(w,2)*u;
        return result;
      }

    template <class Real>
      void  DerivativeCoefficients<Real>::set(const Real & u, const Real & v, const Real & w){
        const Real & denominator = 2.0*pow(w*(u*v-w),3); 
        b[0] = computeC00(u,v,w)/denominator;
        b[1] = computeC01(u,v,w)/denominator;
        b[2] = computeC02(u,v,w)/denominator;
        b[3] = computeC11(u,v,w)/denominator;
        b[4] = computeC12(u,v,w)/denominator;
        b[5] = computeC22(u,v,w)/denominator;
        return;
      }


    template<class Cmplx>
      __device__ __host__
      void accumDerivatives(LinkVariable<Cmplx>* result, const LinkVariable<Cmplx> & left, const LinkVariable<Cmplx> & right, const LinkVariable<Cmplx> & outer_prod)
      {
        Cmplx temp = getTrace(left*outer_prod);
        for(int k=0; k<3; ++k){
          for(int l=0; l<3; ++l){
            result->operator()(k,l) = temp*right(k,l);
          }
        }
        return;
      }


    // Another real hack - fix this!
    template<class T>
      __device__ __host__
    T getAbsMin(const T* const array, int size){
      T min = fabs(array[0]);
      for(int i=1; i<size; ++i){
        T abs_val = fabs(array[i]);
        if((abs_val) < min){ min = abs_val; }   
      }
      return min;
    }

    // What a hack! Yuck!
    template<class Cmplx> 
      __device__ void reciprocalRoot(LinkVariable<Cmplx>* res, DerivativeCoefficients<typename RealTypeId<Cmplx>::Type>* deriv_coeffs, 
																		typename RealTypeId<Cmplx>::Type f[3], const LinkVariable<Cmplx> & q){

        LinkVariable<Cmplx> qsq, tempq;
        qsq = q*q;
        tempq = qsq*q;

        typename RealTypeId<Cmplx>::Type c[3];
        c[0] = getTrace(q).x;
        c[1] = getTrace(qsq).x/2.0;
        c[2] = getTrace(tempq).x/3.0;

        typename RealTypeId<Cmplx>::Type g[3];
        g[0] = g[1] = g[2] = c[0]/3.;
        typename RealTypeId<Cmplx>::Type r,s,theta;
        s = c[1]/3. - c[0]*c[0]/18;
        r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);

        typename RealTypeId<Cmplx>::Type cosTheta = r/sqrt(s*s*s);
        if(fabs(s) < FORCE_UNITARIZE_EPS){
          cosTheta = 1.;
          s = 0.0; 
        }
        if(fabs(cosTheta)>1.0){ r>0 ? theta=0.0 : theta=FORCE_UNITARIZE_PI/3.0; }
        else{ theta = acos(cosTheta)/3.0; }

        s = 2.0*sqrt(s);
        for(int i=0; i<3; ++i){
          g[i] += s*cos(theta + (i-1)*FORCE_UNITARIZE_PI23);
        }

        // New code!
        // Augment the eigenvalues
        // Also need to change q
        typename RealTypeId<Cmplx>::Type delta = getAbsMin(g,3);
        if(delta < HISQ_FORCE_FILTER){
          for(int i=0; i<3; ++i){ 
            g[i]     += HISQ_FORCE_FILTER; 
            q(i,i).x += HISQ_FORCE_FILTER;
          }
          qsq = q*q; // recalculate Q^2
        }
        

        // At this point we have finished with the c's 
        // use these to store sqrt(g)
        for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);

        // done with the g's, use these to store u, v, w
        g[0] = c[0]+c[1]+c[2];
        g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
        g[2] = c[0]*c[1]*c[2];
        
        // set the derivative coefficients!
        deriv_coeffs->set(g[0], g[1], g[2]);

        const typename RealTypeId<Cmplx>::Type & denominator  = g[2]*(g[0]*g[1]-g[2]); 
        c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1]))/denominator;
        c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1])/denominator;
        c[2] =  g[0]/denominator;

        tempq = c[1]*q + c[2]*qsq;
        // Add a real scalar
        tempq(0,0).x += c[0];
        tempq(1,1).x += c[0];
        tempq(2,2).x += c[0];

        f[0] = c[0];
        f[1] = c[1];
        f[2] = c[2];

        if(delta < HISQ_FORCE_FILTER){
          for(int i=0; i<3; ++i){ tempq(i,i).x += HISQ_FORCE_FILTER; }
        }


        res = tempq;
        return;
      }



      // "v" denotes a "fattened" link variable
      template<class Cmplx>
        __device__ __host__
        void getConjugateUnitForceTerm(LinkVariable<Cmplx>* result, const LinkVariable<Cmplx> & v, const LinkVariable<Cmplx> & outer_prod)
        {

          LinkVariable<Cmplx> & local_result = *result;

          LinkVariable<Cmplx> v_dagger   = conj(v);
          LinkVariable<Cmplx> q = v_dagger*v;
          
          
          typename RealTypeId<Cmplx>::Type f[3];
          DerivativeCoefficients<typename RealTypeId<Cmplx>::Type> deriv_coeffs;

          reciprocalRoot<Cmplx>(&rsqrt_q, &deriv_coeffs, f, q);

          LinkVariable<Cmplx> q_vdagger  = q*v_dagger;
          LinkVariable<Cmplx> temp = f[1]*v_dagger + f[2]*qv_dagger;

          local_result = v_dagger*outer_prod*temp;              // result(l,k) =  V^{\dagger}(l,j)*outer_prod(j,i)*temp(i,k)
          local_result += f[2]*qv_dagger*outer_prod*v_dagger;   // result(l,k) += (QV^{\dagger}(l,j)*outer_prod(j,i)*v_dagger(i,k)

          RealTypeId<Cmplx>::Type b[6];
          // Pure hack here
          b[0] = deriv_coeffs.getB00();
          b[1] = deriv_coeffs.getB01();
          b[2] = deriv_coeffs.getB02();
          b[3] = deriv_coeffs.getB11();
          b[4] = deriv_coeffs.getB12();
          b[5] = deriv_coeffs.getB22();


          LinkVariable<Cmplx> qsqv_dagger = q*qv_dagger;
          LinkVariable<Cmplx> pv_dagger   = b[0]*v_dagger + b[1]*qv_dagger + b[2]*qsqv_dagger;
          accumDerivatives(&local_result, v, pv_dagger, outer_prod);

          LinkVariable<Cmplx> rv_dagger = b[1]*v_dagger + b[3]*qv_dagger + b[4]*qsqv_dagger;
          LinkVariable<Cmplx> qv_dagger = q*v_dagger;
          accumDerivatives(&local_result, qv_dagger, rv_dagger, outer_prod);

          LinkVariable<Cmplx> sv_dagger = b[2]*v_dagger + b[4]*qv_dagger + b[5]*qsqv_dagger;
          LinkVariable<Cmplx> qsqv_dagger = q*qv_dagger;
          accumDerivatives(&local_result, qsqv_dagger, sv_dagger, outer_prod);

          return;
        }


      // "v" denotes a "fattened" link variable
      template<class Cmplx>
        __device__ __host__
        void getUnitForceTerm(LinkVariable<Cmplx>* result, const LinkVariable<Cmplx> & v, const LinkVariable<Cmplx> & outer_prod)
        {
          typename RealTypeId<Cmplx>::Type f[3]; 
          typename RealTypeId<Cmplx>::Type b[6];

          LinkVariable<Cmplx> v_dagger = conj(v);  // okay!
          LinkVariable<Cmplx> q   = v_dagger*v;    // okay!

          LinkVariable<Cmplx> rsqrt_q;

          DerivativeCoefficients<typename RealTypeId<Cmplx>::Type> deriv_coeffs;

          // Yuck!
          reciprocalRoot<Cmplx>(&rsqrt_q, &deriv_coeffs, f, q);

          // Pure hack here
          b[0] = deriv_coeffs.getB00();
          b[1] = deriv_coeffs.getB01();
          b[2] = deriv_coeffs.getB02();
          b[3] = deriv_coeffs.getB11();
          b[4] = deriv_coeffs.getB12();
          b[5] = deriv_coeffs.getB22();

          LinkVariable<Complex> & local_result = *result;
          local_result = rsqrt_q*outer_prod;

          // We are now finished with rsqrt_q
          LinkVariable<Cmplx> qv_dagger  = q*v_dagger;
          LinkVariable<Cmplx> vv_dagger  = v_dagger*v; // Not necessarily equal to conj(q) because of the cutoff
          LinkVariable<Cmplx> vqv_dagger = v*qv_dagger;
          LinkVariable<Cmplx> temp = f[1]*vv_dagger + f[2]*vqv_dagger;

          local_result += outer_prod*temp + f[2]*q*outer_prod*vv_dagger;



          // now done with vv_dagger, I think
          LinkVariable<Cmplx> qsqv_dagger = q*qv_dagger;
          LinkVariable<Cmplx> pv_dagger   = b[0]*v_dagger + b[1]*qv_dagger + b[2]*qsqv_dagger;
          accumDerivatives(&local_result, v, pv_dagger, outer_prod);
          
          LinkVariable<Cmplx> rv_dagger = b[1]*v_dagger + b[3]*qv_dagger + b[4]*qsqv_dagger;
          LinkVariable<Cmplx> vq = v*q;
          accumDerivatives(&local_result, vq, rv_dagger, outer_prod);

          LinkVariable<Cmplx> sv_dagger = b[2]*v_dagger + b[4]*qv_dagger + b[5]*qsqv_dagger;
          LinkVariable<Cmplx> vqsq = vq*q;
          accumDerivatives(&local_result, vqsq, sv_dagger, outer_prod);

          return;
        } // get unit force term


/* 
       template<class Cmplx>
         __device__ __host__
         void getCompleteUnitForceTerm(LinkVariable<Cmplx>* result, const LinkVariable<Cmplx> & y, const LinkVariable<Cmplx> & outer_prod)
         {
            LinkVariable<Cmplx> v_dagger = conj(v);
            LinkVariable<Cmplx> q = v_dagger*v;
            LinkVariable<Cmplx> rsqrt_q;

            LinkVariable<Complex> & local_result = *result;
            local_result = rsqrt_q*outer_prod;

            LinkVariable<Cmplx> qv_dagger = q*v_dagger;
            LinkVariable<Cmplx> vv_dagger = conj(q); // true up to the cutoff contribution
            LinkVariable<Cmplx> vqv_dagger = v*qv_dagger;

            LinkVariable<Cmplx> temp = f[1]*vv_dagger + f[2]*vqv_dagger;

            local_result += outer_prod*temp + f[2]*q*outer_prod*vv_dagger;

             LinkVariable<Cmplx> qsqv_dagger = q*qv_dagger;
             LinkVariable<Cmplx> pv_dagger = b[0]*v_dagger + b[1]*qv_dagger + b[2]*qsqv_dagger;
             accumDerivatives(&local_result, v, pv_dagger, outer_prod);
           
             // Need to add the bits for the parts of the force contribution coming from W^{\dagger} 
                
             // Need to double-check Matrix derivatives!

             return;
         }
*/

  } // namespace fermion_force
} // namespace hisq
