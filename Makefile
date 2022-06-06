gen:
python gen_graycode_imgs.py 1080 1920 -graycode_step 2

cap:
python cap_chessboard.py

calib:
python calibrate.py 1080 1920 9 7 0.024 2 -black_thr 40 -white_thr 5 camera_config.json

clean:
rm -rf capture_*