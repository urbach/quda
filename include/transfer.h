#ifndef _TRANSFER_H
#define _TRANSFER_H

/**
 * @file transfer.h
 *
 * @section DESCRIPTION 
 *
 * Defines the prolongation and restriction operators used to transfer
 * between grids.
 */

#include <color_spinor_field.h>

class Transfer {

  /** The block-normalized null-space components that define the prolongator */
  cpuGaugeField V;

  /** The mapping onto coarse sites from fine sites */
  unsigned int *geo_map;

  /** The mapping onto coarse spin from fine spin */
  unsigned int *spin_map;

  /** A temporary field with fine geometry but coarse color */
  cpuColorSpinorField tmp;

  /** 
   * The constructor for Transfer
   * @param B Array of null-space vectors
   * @param Nvec Number of null-space vectors
   * @param geo_map Geometric mapping from fine grid to coarse grid 
   * @param spin_map Mapping from fine spin to coarse spin 
   */
  Transfer(cpuColorSpinorField *B, int Nvec, int *geo_bs, int spin_bs) {

  /** The destructor for Transfer */
  virtual ~Transfer();

  /** 
   * Apply the prolongator
   * @param out The resulting field on the fine lattice
   * @param in The input field on the coarse lattice
   */
  void P(cpuColorSpinorField &out, const cpuColorSpinorField &in);

  /** 
   * Apply the restrictor 
   * @param out The resulting field on the coarse lattice
   * @param in The input field on the fine lattice   
   */
  void R(cpuColorSpinorField &out, const cpuColorSpinorField &in);

};


#endif // _TRANSFER_H