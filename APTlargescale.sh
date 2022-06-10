#!/bin/bash

for i in {1..100}
do
    if (( $i % 6 == 0 )) # every 6th execution do not use '&' to run next script. This way only 6 cpus will be used at once.
    then   
        python APT /data1/people/jdtaylor/fake_data/fake_$i.par \
/data1/people/jdtaylor/fake_data/fake_$i.tim --data_path /data1/people/jdtaylor --plot_final False\

    else
        python APT /data1/people/jdtaylor/fake_data/fake_$i.par \
/data1/people/jdtaylor/fake_data/fake_$i.tim --data_path /data1/people/jdtaylor --plot_final False &

    fi
    if (( $i % 12 == 0 )) # do not want to accidentally have every 6th execution run particilarly fast, thus every twelth it must wait until every process is done
    then
    wait
    fi
done
wait 
echo "done"

date