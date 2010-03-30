#!/bin/bash

DSLASH=./dslash_test

function test_prec_recon(){
  precisions="double single half"
  recons="12 8"
  types="0 1 2"
  for sprec in $precisions
  do
	for gprec in $precisions
	do 
		for recon in $recons
		do
			for type in $types
			do
			cmd="${DSLASH} --sprec $sprec --gprec $gprec --recon $recon --type $type"
			echo $cmd
			$cmd
			cmd="${DSLASH} --sprec $sprec --gprec $gprec --recon $recon --type $type --dagger"
			echo $cmd
			$cmd
			done
		done #recon
	done #gprec
  done #sprec

}

function test_t_dim()
{ 
  precisions="double single half"
  recons="12 8"
  tdims="4 8 16 32 48 64 80 96 112 128"
  for prec in $precisions
  do
        for recon in $recons
        do 
		for tdim in $tdims
		do
             		cmd="${DSLASH} --sprec $prec --gprec $prec --recon $recon --tdim $tdim"
             		echo $cmd
             		$cmd
		done #tdim
        done #recons
  done #prec
}

test_t_dim

