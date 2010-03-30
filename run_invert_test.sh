#!/bin/bash

ret_int="0"
precision_int()
{
	prec=$1
	if [ $prec = "double" ] ; then
		ret_int="0"
	fi

   	if [ $prec = "single" ]; then
		ret_int="1"
	fi

	if [ $prec = "half" ]; then
		ret_int="2"
	fi
	
}


function test_prec_recon_type(){
  DSLASH=./invert_test
  precisions="double single half"
  recons="12 8"
  types="0 1 2"

  for sprec in $precisions
  do
	for gprec in $precisions
	do 
		for recon in $recons
		do
			for sprec_sloppy in $precisions
			do
				for gprec_sloppy in $precisions
				do 
					for recon_sloppy in $recons
					do			
					for type in $types
					do
						precision_int $sprec					
						sprec_int=$ret_int	
						precision_int $sprec_sloppy
						sprec_sloppy_int=$ret_int	
						if [ $sprec_int -gt $sprec_sloppy_int ]; then
							continue
						fi

                                                precision_int $gprec
                                                gprec_int=$ret_int
                                                precision_int $gprec_sloppy
                                                gprec_sloppy_int=$ret_int
                                                if [ $gprec_int -gt $gprec_sloppy_int ]; then
                                                        continue
                                                fi

                                                if [ $recon_sloppy -gt $recon ]; then
                                                        continue
                                                fi

						cmd="${DSLASH} --sprec $sprec --gprec $gprec --recon $recon --sprec_sloppy $sprec_sloppy --gprec_sloppy $gprec_sloppy --recon_sloppy $recon_sloppy --type $type "
						echo 
						echo $cmd
						$cmd
					done #type
					done
				done
			done
		done
	done #gprec
  done #sprec
}


function test_t_size()
{
  prog=./invert_test
  precisions="double single half"
  recons="12 8"
  tdims="4 8 16 32 48 64 80 96 112 128"
  for prec in $precisions
  do 
	for recon in $recons
	do
		for tdim in $tdims
		do
			cmd="${prog} --sprec $prec --gprec $prec --recon $recon  --tdim $tdim"
			echo $cmd
			$cmd
		done
	done
  done
}

test_t_size

