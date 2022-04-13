#!/bin/bash
# Author:Anna
# Date & Time: 2021-02-12 18:27:25
# Description: cut tremor events from continuous data for each 60-s

cd ~

temp1=file_fold.txt

ls -d *WS_20* > $temp1
#echo WS_2015_01_03 > $temp1

N_loop=`gawk '{print NR}' $temp1`

for n in $N_loop
do
    foldname=`gawk '{if(NR=='$n')print $1}' $temp1`

    home_path=/home/anna/$foldname

    cd $home_path

    echo "start $home_path"

    mkdir Cut_file

    temp2=filename.txt

    ls *.vel* > $temp2

    L_loop=`gawk '{print NR}' $temp2`
    for l in $L_loop
    do
        filename=`gawk '{if(NR=='$l')print $1}' $temp2`

        i=0
        j=80
    
        while [ "$j" -lt "86400" ] #less than

        do
            echo "cuterr fillz ; cut $i $j; r $filename; w append .cut.$i ; q " | sac
            #echo " cuterr fillz ; cut $cut_st $cut_et ;r *$m_jday*.SAC ; bp c 2 8 ; w append .$event.bp ; q "|sac
            i=$(($i+60))
            j=$[$j+60]
        done

        
        
        echo "start mv"

        mv *cut* Cut_file
    done

    cd Cut_file

    mkdir E N U

    mv *.E.SAC* E

    mv *.N.SAC* N

    mv *.U.SAC* U

    cd ..

    rm $temp2

    cd ~

done

rm $temp1

echo "Finish!"
