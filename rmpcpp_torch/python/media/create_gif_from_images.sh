width=900
height=1016
x=500
y=0

ffmpeg -framerate 200 -start_number 0 -i frame_%05d.png -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" out.gif

# For non-looped:

#ffmpeg -framerate 200 -start_number 0 -i frame_%05d.png -vf "fps=30, crop=w=${width}:h=${height}:x=${x}:y=${y}, split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop -1 out.gif
