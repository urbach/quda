#ifndef _TUNE_QUDA_H
#define _TUNE_QUDA_H

#include <quda_internal.h>
#include <dirac_quda.h>

// Curried wrappers to Cuda functions used for auto-tuning
class TuneBase {

 protected:
  const char *name;
  QudaVerbosity verbose;

 public:
  TuneBase(const char *name, QudaVerbosity verbose) : 
    name(name), verbose(verbose) { ; }
   
  virtual ~TuneBase() { ; }
  virtual void Apply() const = 0;
  virtual void ApplyMulti(int i) const = 0;
  virtual unsigned long long Flops() const = 0;
 
  const char* Name() const { return name; }

  // Varies the block size of the given function and finds the performance maxiumum
  void Benchmark(dim3 &block); 
  void BenchmarkMulti(dim3 * block, int n);
};

class TuneDiracWilsonDslash : public TuneBase {

 private:
  const DiracWilson &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracWilsonDslash(const DiracWilson &d, cudaColorSpinorField &a, 
			const cudaColorSpinorField &b) : 
  TuneBase("DiracWilsonDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracWilsonDslash() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DiracWilson::Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracWilsonDslashXpay : public TuneBase {

 private:
  const DiracWilson &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracWilsonDslashXpay(const DiracWilson &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracWilsonDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracWilsonDslashXpay() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DiracWilson::DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracClover : public TuneBase {

 private:
  const DiracClover &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracClover(const DiracClover &d, cudaColorSpinorField &a, 
		  const cudaColorSpinorField &b) : 
  TuneBase("DiracClover", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracClover() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.Clover(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracCloverDslash : public TuneBase {

 private:
  const DiracCloverPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracCloverDslash(const DiracCloverPC &d, cudaColorSpinorField &a, 
		  const cudaColorSpinorField &b) : 
  TuneBase("DiracCloverDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracCloverDslash() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracCloverDslashXpay : public TuneBase {

 private:
  const DiracCloverPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracCloverDslashXpay(const DiracCloverPC &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracCloverDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracCloverDslashXpay() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMass : public TuneBase {

 private:
  const DiracTwistedMass &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracTwistedMass(const DiracTwistedMass &d, cudaColorSpinorField &a, 
		       const cudaColorSpinorField &b) : 
  TuneBase("DiracTwistedMass", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracTwistedMass() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.Twist(a, b); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMassDslash : public TuneBase {

 private:
  const DiracTwistedMassPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracTwistedMassDslash(const DiracTwistedMassPC &d, cudaColorSpinorField &a, 
			     const cudaColorSpinorField &b) : 
  TuneBase("DiracTwistedMassDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracTwistedMassDslash() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMassDslashXpay : public TuneBase {

 private:
  const DiracTwistedMassPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracTwistedMassDslashXpay(const DiracTwistedMassPC &d, cudaColorSpinorField &a, 
				 const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracTwistedMassDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracTwistedMassDslashXpay() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracDomainWallDslash : public TuneBase {

 private:
  const DiracDomainWall &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracDomainWallDslash(const DiracDomainWall &d, cudaColorSpinorField &a, 
			    const cudaColorSpinorField &b) : 
  TuneBase("DiracDomainWallDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracDomainWallDslash() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracDomainWallDslashXpay : public TuneBase {

 private:
  const DiracDomainWall &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracDomainWallDslashXpay(const DiracDomainWall &d, cudaColorSpinorField &a, 
				const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracDomainWallDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracDomainWallDslashXpay() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracStaggeredDslash : public TuneBase {

 private:
  const DiracStaggered &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracStaggeredDslash(const DiracStaggered &d, cudaColorSpinorField &a, 
			const cudaColorSpinorField &b) : 
  TuneBase("DiracStaggeredDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracStaggeredDslash() { ; }

  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DiracStaggered::Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracStaggeredDslashXpay : public TuneBase {

 private:
  const DiracStaggered &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
 TuneDiracStaggeredDslashXpay(const DiracStaggered &d, cudaColorSpinorField &a, 
			      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracStaggeredDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracStaggeredDslashXpay() { ; }
  
  void ApplyMulti(int i) const {}
  void Apply() const { dirac.DiracStaggered::DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

#ifdef GPU_FATLINK
#include "llfat_quda.h"
class TuneLinkFattening : public TuneBase {  
 private:
  FullGauge& cudaFatLink;
  FullGauge& cudaSiteLink;
  FullStaple& cudaStaple;
  FullStaple& cudaStaple1;  
  llfat_kernel_param_t & kparam;
  llfat_kernel_param_t & kparam_1g;

 public:  
 TuneLinkFattening(FullGauge& _cudaFatLink, FullGauge& _cudaSiteLink,
		   FullStaple& _cudaStaple, FullStaple& _cudaStaple1,
		   llfat_kernel_param_t _kparam,
		   llfat_kernel_param_t _kparam_1g,
		   QudaVerbosity verbose) : 
  TuneBase("TuneLinkFattening", verbose), cudaFatLink(_cudaFatLink), 
    cudaSiteLink(_cudaSiteLink), cudaStaple(_cudaStaple), cudaStaple1(_cudaStaple1),
    kparam(_kparam), kparam_1g(_kparam_1g){ }
 virtual ~TuneLinkFattening() {}
 
 void Apply() const {}
 void ApplyMulti(int idx) const {
   switch(idx){
   case 0:
     siteComputeGenStapleParityKernel_ex((void*)cudaStaple.even, (void*)cudaStaple.odd,
					 (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					 (void*)cudaFatLink.even, (void*)cudaFatLink.odd,
					 0, 1,
					 0.01,
					 cudaSiteLink.reconstruct, 
					 cudaSiteLink.precision, kparam_1g);
       
       break;
   case 1:
     computeGenStapleFieldParityKernel_ex((void*)NULL, (void*)NULL,
					  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					  (void*)cudaFatLink.even, (void*)cudaFatLink.odd,
					  (void*)cudaStaple.even, (void*)cudaStaple.odd,
					  0, 1, 0,/*this 0 means save_staple=0*/
					  0.01,
					  cudaSiteLink.reconstruct, 
					  cudaSiteLink.precision,
					  kparam);
     
     break;
   case 2:
     computeGenStapleFieldParityKernel_ex((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
					  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					  (void*)cudaFatLink.even, (void*)cudaFatLink.odd,
					  (void*)cudaStaple.even, (void*)cudaStaple.odd,
					  0, 1, 1,/*this 1 means save_staple=1*/
					  0.01,
					  cudaSiteLink.reconstruct, 
					  cudaSiteLink.precision,
					  kparam_1g);
     break;
   default:
     errorQuda("Wrong func idx(%d)\n", idx);
   }
 }
 unsigned long long Flops() const { return 0;};
};

#endif


#endif // _TUNE_QUDA_H
