python test_video.py \
  --model_path_i    ./src/checkpoints/cvpr2025_image.pth.tar \
  --model_path_p    ./src/checkpoints/cvpr2025_video.pth.tar \
  --rate_num        64 \
  --test_config     ./dataset_config.json \
  --cuda            1 \
  -w                1 \
  --write_stream    1 \
  --force_zero_thres 0.12 \
  --output_path     all_q_re_64.json \
  --force_intra_period -1 \
  --reset_interval  64 \
  --force_frame_num -1 \
  --check_existing  0 \
  --verbose         1

