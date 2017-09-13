#!/bin/bash

for dir in $(ls -d */); do
    cd $dir
    echo $dir
    if [[ ! -f ssim.log ]]; then
	    /video-drive1-local/jemmons/convert.sh beforeFile.raw beforeFile.y4m &
	    /video-drive1-local/jemmons/convert.sh afterFile.raw afterFile.y4m &
	    wait
	    
	    dump_ssim -p 8 beforeFile.y4m afterFile.y4m | tee ssim.log
    fi
    cd ..
done
