#!/bin/bash

name=GAN_ellipse_gpu6_1-8-23

swif2 create -workflow ${name}

N_epochs=500000      #Number of epcohs for training
workdir=/work/clas12/reedtg/data_science/GAN_ellipse_example/
#output_file_name=an


swif2 add-job -workflow ${name} \
    	-constraint gpu \
    	-account clas12 \
    	-partition gpu \
    	-disk 1gb \
	-ram 1gb \
	-tag key val \
    	-time 1days \
	"source /group/clas12/packages/setup.sh && module load anaconda3 && cd $workdir && \
                       source /apps/anaconda3/2023.03/bin/activate ds_env && ellipse_GAN.py ${N_epochs}"

swif2 run -workflow ${name}


#For running on CPU:
#-constraint general
#-partition production
#-disk 1gb
#-ram 1gb

#For running on GPU:
#-constraint gpu
#-partition gpu
