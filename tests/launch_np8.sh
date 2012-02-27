mpirun -np 8 -machinefile $PBS_NODEFILE ./domain_wall_invert_test --prec single --prec_sloppy half --tgridsize 8

#mpirun -np 8 -machinefile $PBS_NODEFILE ./domain_wall_invert_test --prec single --prec_sloppy half --ygridsize 2 --zgridsize 2 --tgridsize 2
