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
done < sorted.txt  > finaloutput.txt
