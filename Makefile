gen:
	python gen_graycode_imgs.py 1080 1920

cap:
	python cap_chessboard.py

cal:
	python calibrate.py 1080 1920 10 7 0.055 1 -camera camera_config.json

clean:
	rm -rf captures/*
