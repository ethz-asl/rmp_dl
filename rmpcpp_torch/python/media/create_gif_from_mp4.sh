#!/bin/bash

# For the rotating world gifs
#width=980
#height=980
#x=450
#y=40

# For the ray loop gifs
width=900
height=1016
x=500
y=0


# speedup factor is the 2nd argument if given, or otherwise 5x
sf=${2-5}

# Using find to locate mp4 files and process them
find "$1" -type f -name "*.mp4" ! -name "*_temp.mp4" -print0 | while IFS= read -r -d '' file
do
	if [ -f "$file" ]; then
		base=$(basename "$file" .mp4)
		dname=$(dirname "$file")
		echo "Processing: ${file}"
		echo "Dname: ${dname}"
		echo "Base: ${base}"

		# Step 1: Speed up the video and save as a temporary file
		echo "Speeding up mp4"
		ffmpeg -i "${file}" -vf "crop=w=${width}:h=${height}:x=${x}:y=${y}, setpts=PTS/${sf}" -an "${dname}/${base}_temp.mp4"  

		# Step 2: Convert the sped-up video to GIF
		echo "Converting sped up mp4 to gif"
		ffmpeg -nostdin -y -i "${dname}/${base}_temp.mp4" \
			-vf "fps=30, scale=-1:600:flags=lanczos, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
			-loop 0 \
			"${dname}/${base}_output_${sf}.gif"

		# Remove the temporary file
		echo "Removing mp4\n\n\n\n"
		rm "${dname}/${base}_temp.mp4"
	else
		echo "File not found: $file"
	fi   

done
