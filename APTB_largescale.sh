#!/bin/bash
PRESENT=$PWD
cd /data1/people/jdtaylor/binary_fake_data6/
for i in {1..100}
do
    if (( $i % 6 == 0 )) # every 6th execution do not use '&' to run next script. This way only 6 cpus will be used at once.
    then   
        APT_binary fake_$i.par fake_$i.tim --plot_final False

    else
        APT_binary fake_$i.par fake_$i.tim --plot_final False &

    fi
    # if (( $i % 12 == 0 )) # do not want to accidentally have every 6th execution run particilarly fast, thus every twelth it must wait until every process is done
    # then
    # wait
    # fi
done
wait 
echo "done"
cd $PRESENT

date