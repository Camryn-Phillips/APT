#!/bin/bash

for i in {1..100}
do
    if (( $i % 5 == 0 )) # every 5th execution do not use '&' to run next script. This way only 5 cpus will be used at once.
    then   
        python simdata.py

    else
        python simdata.py &

    fi
done
wait 
echo "done"

date