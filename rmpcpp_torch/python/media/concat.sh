#!/bin/bash

counter=0
temp_files=()

find $1 -name "*.mp4" > video_list.txt

while read -r video_path; do
	echo $video_path
	video_name=$(basename "${video_path}")
	temp_file="temp_${counter}.mp4"
	temp_files+=($temp_file)

	ffmpeg -nostdin -hide_banner -loglevel panic -i "$video_path" -vf "drawtext=text='${video_name}':x=10:y=10:fontsize=24:fontcolor=white" -c:a copy $temp_file

	counter=$((counter+1))
done < "video_list.txt"

rm video_list.txt

ffmpeg -f concat -safe 0 -i <(for f in "${temp_files[@]}"; do echo "file '$PWD/$f'"; done) -c copy output.mp4

rm -f "$(temp_files[@]}"
