#!/bin/bash
# Author:Anna
# Date & Time: 2021-03-17 09:53:22
# Description: make tremor list

#home_path=~/Western_Shikoku_Data_temp
home_path=~/Data_Western_Shikoku
cut_file=Cut_file/E
list_fold=~/List_Western_Shikoku

cd $home_path

temp0=file_fold.txt
ls -d *WS-w_20* > $temp0


N_loop=`gawk '{print NR}' $temp0`
for n in $N_loop
do
	foldname=`gawk '{if(NR=='$n')print $0}' $temp0`
	fold_path=$home_path/$foldname
	cd $fold_path
	cd $cut_file
	ls * > temp.txt
	sort -t. -k8 -n temp.txt > $foldname.txt
	rm temp.txt
	mv *.txt $list_fold
	cd $home_path
done

rm $temp0