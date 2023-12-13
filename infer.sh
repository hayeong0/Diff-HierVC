python3 inference.py \
    --src_path './sample/src_p241_004.wav' \
    --trg_path './sample/tar_p239_022.wav' \
    --ckpt_model './ckpt/model_diffhier.pth' \
    --ckpt_voc './vocoder/voc_ckpt.pth' \
    --output_dir './converted' \
    --diffpitch_ts 30 \
    --diffvoice_ts 6
