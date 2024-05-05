width=900
height=1016
x=500
y=0



ffmpeg -y -framerate 30 -start_number 0  -i "$1/frame_%05d.png" -vframes 1 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop -1 "$1/00initial_frame.gif" 

ffmpeg -y -framerate 150 -start_number 0  -i "$1/frame_%05d.png" -vframes 51 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop -1 "$1/10initial.gif"

ffmpeg -y -framerate 90 -start_number 253 -i "$1/frame_%05d.png" -vframes 1 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop -1 "$1/19initial_freeze.gif"

ffmpeg -y -framerate 90 -start_number 253 -i "$1/frame_%05d.png" -vframes 67 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop -1 "$1/20rays.gif"

ffmpeg -y -framerate 200 -start_number 451 -i "$1/frame_%05d.png" -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop -1 "$1/30runout.gif"



# LOOPED BELOW
#ffmpeg -framerate 30 -start_number 0  -i "$1/frame_%05d.png" -vframes 1 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" looped-00initial_frame.gif
#
#ffmpeg -framerate 150 -start_number 0  -i "$1/frame_%05d.png" -vframes 51 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" looped-0initial.gif
#
#ffmpeg -framerate 90 -start_number 253 -i "$1/frame_%05d.png" -vframes 67 -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" looped-1rays.gif
#
#ffmpeg -framerate 200 -start_number 451 -i "$1/frame_%05d.png" -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" looped-2other.gif
