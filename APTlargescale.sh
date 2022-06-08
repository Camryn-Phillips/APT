#!/bin/bash

for i in {1..101}
do
    if (( $i % 5 == 0 )) # every 5th execution do not use '&' to run next script. This way only 5 cpus will be used at once.
    then   
        python APT.py /data1/people/jdtaylor/fake_data/fake_$i.par \
/data1/people/jdtaylor/fake_data/fake_$i.tim --data_path /data1/people/jdtaylor --plot_final False\

    else
        python APT.py /data1/people/jdtaylor/fake_data/fake_$i.par \
/data1/people/jdtaylor/fake_data/fake_$i.tim --data_path /data1/people/jdtaylor --plot_final False &

    fi
done
wait echo "done"

date