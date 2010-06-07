
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


function test_prec_recon_type()
{
    INVERTER=./staggered_invert_test
    precisions="double single half"
    recons="18 12 8"
    tests="0 3"
    
    for prec in $precisions
    do
	for recon in $recons
	do
            for prec_sloppy in $precisions
            do
		for recon_sloppy in $recons
		do
                    for test in $tests
                    do
			precision_int $prec
			prec_int=$ret_int
			precision_int $prec_sloppy
			prec_sloppy_int=$ret_int
			if [ $prec_int -gt $prec_sloppy_int ]; then
                            continue
			fi
			
			
			if [ $recon_sloppy -gt $recon ]; then
                            continue
			fi
			
			cmd="${INVERTER} --prec $prec --recon $recon --prec_sloppy $prec_sloppy --recon_sloppy $recon_sloppy --test $test "
			echo "--------------------------------------------------------------------"
			echo $cmd
			$cmd
                    done #type
		done
            done
	done
    done 
}

make all
test_prec_recon_type
