#this script is used to split the video of any format into parts depending on number of nodes in cluster

#Usage -  bash ffmpeg.sh <filename> <no_nodes>

#!/bin/sh


# used to take command line argument as video file and store it into varible	

#special variables

FILE_NAME=$1
FILE_NAME_IMG=$2
echo $FILE_NAME
echo $FILE_NAME_IMG
mv "$FILE_NAME_IMG" /tmp/reference.jpg
#THRESHOLD=$3
#echo $THRESHOLD >> threshold.txt


hdfs dfs -mkdir /home/hduser/inputData
#hdfs dfs -put threshold.txt /home/hduser/inputData
hdfs dfs -put /tmp/reference.jpg /home/hduser/inputData


# used to store duration of video file
#convert duration HH:MM:SS in seconds
DURATION=$(avconv -i $FILE_NAME 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, ":"); split(A[3], B, "."); print 3600*A[1] + 60*A[2] + B[1] }')

#splitting a video
START_TIME=0

#calculate file size

FILESIZE=$(stat -c%s "$FILE_NAME")

DEFAULT="104857600" #104857600
let NO_NODES=FILESIZE/DEFAULT
#Change time dynamically 

# division value used

let DIFF=DURATION/NO_NODES
COUNTER=0 

#create a dir
part1=`dirname $FILE_NAME`
part2=`basename $FILE_NAME`
mkdir /tmp/$part2

# while loop basically used to store splitted videos in to /data/respective folder

while [  $COUNTER -lt $NO_NODES ]; 
	do	
		echo $START_TIME,$DIFF
		echo $THRESHOLD >> threshold.txt
             	avconv -ss $START_TIME -i $FILE_NAME -t $DIFF  -vcodec copy -acodec copy "/tmp/"$part2"/demo"$COUNTER".mp4"
		let START_TIME=START_TIME+DIFF	
		let COUNTER=COUNTER+1 
         done
let START_TIME=START_TIME-DIFF
let COUNTER=COUNTER-1 
if [ ! -e /tmp/timer.txt ]; then
	echo >> /tmp/timer.txt
fi
echo $COUNTER >> /tmp/timer.txt
echo $START_TIME >> /tmp/timer.txt
echo $DURATION
echo $DIFF
echo $FILESIZE,$NO_NODES

hadoop fs -put /tmp/timer.txt /home/hduser/inputData

# end of ffmpeg 

# switch to hduser 

bash /home/hduser/finalcode3/split_map/hdfs.sh $NO_NODES $FILE_NAME

