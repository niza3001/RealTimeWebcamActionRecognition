#!/bin/bash

#go into dataset directory
cd Dataset2
gnum=1
cnum=1
g=25
c=7


#for each class
for class in */ ; do
	echo "$class"
	# for each file in class
	cd "$class"
	for f in *.mp4 ; do
		#mv "$f" v_"$class"_g01_v01.mp4
		className=$(echo "$class" | cut -f 1 -d '/')
		new="v_${className}_g$(printf %02d $gnum)_c$(printf %02d $cnum).mp4"
		echo "$new"
		name=$(echo "$new" | cut -f 1 -d '.')
		echo "$name"
		if [[ -d "../$name" ]]; then
			echo "Directory exists at $name"
		else
    		mkdir "../$name"
		fi
		#cd "$name"

		echo "$f"
		ffmpeg -i "$f" -vf scale=342:256 -strict -2 "$new" -hide_banner
		ffmpeg -i "$new" -vf fps=10 ../"$name"/frame%06d.jpg -hide_banner
		gnum=$((gnum+1))
		if [ "$gnum" -eq "$g" ];then
  			gnum=1
  			cnum=$((cnum+1))
		fi
	done
	gnum=1
	cnum=1
	cd ..
done


