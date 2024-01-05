#!/bin/bash

name=GAN_ellipse_test_1-5-23

swif2 create -workflow ${name}

N_jobs=100      #Number of jobs
workdir=/work/clas12/reedtg/data_science/GAN_ellipse_example/
#output_file_name=an
#N_loops=10
#counts_max=500000
#counts_max=1000

#for ((i = 1; i <= $N_jobs; i++))
#do
	swif2 add-job -workflow ${name} \
    		-constraint general \
    		-account clas12 \
    		-partition production \
    		-tag key val \
    		-disk 1gb \
    		-ram 1gb \
    		-time 1days \
		"source /group/clas12/packages/setup.sh && module load anaconda3 && cd $workdir && \
                        source /apps/anaconda3/2023.03/bin/activate ds_env && ellipse_GAN.py"
#done

swif2 run -workflow ${name}
