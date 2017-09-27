#!/bin/sh

#script to upload video to hdfs

NO_NODES=$1
FILE_NAME=$2
COUNTER=0

#create a directory 
part1=`dirname $FILE_NAME`
part2=`basename $FILE_NAME`
arg=$part2
beforedot=${arg%.*}

hdfs dfs -mkdir /home/hduser/$beforedot

while [  $COUNTER -lt $NO_NODES ]; 
	do	
		hdfs dfs -put "/tmp/"$part2"/demo"$COUNTER".mp4" /home/hduser/$beforedot
		let COUNTER=COUNTER+1
	done

part2=$part2 | cut -d '.' -f1
hadoop jar /home/hduser/finalcode3/mapreduce/wc.jar DirectVideoProcessor /home/hduser/$beforedot /home/hduser/$beforedot"output"
hadoop fs -get /home/hduser/$beforedot"output"
grep -v 'demo' part-r-00000 > output.txt
sort -t\: -k 1,1n -k 2,2n -k 3,3n output.txt > sorted.txt
grep '[^[:blank:]]' < sorted.txt > input_sorted.txt
count_occurance=0
previous_bc=0
while read line           
do  
	hr="$( cut -d ':' -f 1 <<< "$line" )"
  	min="$( cut -d ':' -f 2 <<< "$line" )"
	sec="$( cut -d ':' -f 3 <<< "$line" )"
	let totalsec=hr*3600+min*60+sec
	let bc=totalsec/300
	if [  $bc == $previous_bc ];
	then
    	   let count_occurance=count_occurance+1
	else
	   let count_occurance=count_occurance+1
           echo $hr":"$min":00" $count_occurance
	   let count_occurance=0
	   #echo $count_occurance
	   let previous_bc=bc
	fi	           
done < <(tr -d '\r' < input_sorted.txt)  > finaloutput.txt  




