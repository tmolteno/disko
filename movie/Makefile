HDR=bayes_tart_2019_08_04_21_38_31_UTC
FFMPEG=ffmpeg -i ${HDR}_s%05d.png

data:
	disko_bayes --fov 155 --ms ../test_data/test.ms  --PNG --arcmin=120  --nsamples 1000 --title 'bayes_tart' --sigma-v=0.1

movie:
	${FFMPEG} video.webm
	
mov2:
	${FFMPEG} -c:v libvpx-vp9 -b:v 0 -crf 30 -pass 1 -an -f null /dev/null && \
	${FFMPEG} -c:v libvpx-vp9 -b:v 0 -crf 30 -pass 2 -c:a libopus output.webm

	 
clean:
	rm -f *.png *.webm
