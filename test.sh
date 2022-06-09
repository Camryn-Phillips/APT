#!/bin/bash

for i in 35 36 37 37 37 40
do
    if (( $i % 5 == 0 ))
    then   
        echo $i
    else
        echo "$i is not div by 5"
    fi
done
wait
echo "done"

date