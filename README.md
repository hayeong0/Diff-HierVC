##  Diff-HierVC: Diffusion-based Hierarchical Voice Conversion with Robust Pitch Generation and Masked Prior for Zero-shot Speaker Adaptation  <a src="https://img.shields.io/badge/cs.CV-2311.04693-b31b1b?logo=arxiv&logoColor=red" href="http://arxiv.org/abs/2311.04693"> <img src="https://img.shields.io/badge/cs.CV-2311.04693-b31b1b?logo=arxiv&logoColor=red">

The official Pytorch implementation of Diff-HierVC (Interspeeh 2023, Oral)
   
Ha-Yeong Choi, Sang-Hoon Lee, Seong-Whan Lee 

![image](https://github.com/hayeong0/Diff-HierVC/assets/47182864/e8a22c5f-af6f-43e8-92b0-0aac839cb0b6)
<p align="center"><em> Overall architecture </em>

> Although voice conversion (VC) systems have shown a remarkable ability to transfer voice style, existing methods still have an inaccurate pitch and low speaker adaptation quality. To address these challenges, we introduce Diff-HierVC, a hierarchical VC system based on two diffusion models. We first introduce DiffPitch, which can effectively generate $F_0$ with the target voice style. Subsequently, the generated $F_0$ is fed to DiffVoice to convert the speech with a target voice style. Furthermore, using the source-filter encoder, we disentangle the speech and use the converted Mel-spectrogram as a data-driven prior in DiffVoice to improve the voice style transfer capacity. Finally, by using the masked prior in diffusion models, our model can improve the speaker adaptation quality. Experimental results verify the superiority of our model in pitch generation and voice style transfer performance, and our model also achieves a CER of 0.83\% and EER of 3.29\% in zero-shot VC scenarios.

## 🎧 Audio Demo
https://diff-hiervc.github.io/audio_demo/

## 📑 Pre-trained Model
Our model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1THkeyDlA7EbZxwnuuxGsUOftV70Fb7h4?usp=sharing).

- model_diffhier.pth
- voc_hifigan.pth
- voc_bigvgan.pth

## 🔨 Usage

1. Clone this rep && Install python requirement

```
git clone https://github.com/hayeong0/Diff-HierVC.git
pip install -r req*
```

2. Download the pre-trained model checkpoint from drive and place it in the following path.
```
.
├── ckpt
│   ├── config.json
│   └── model_diffhier.pth ✅
├── inference.py
├── infer.sh
├── model
├── module
├── requirements.txt
├── utils
└── vocoder
    ├── hifigan.py
    ├── modules.py
    └── voc_hifigan.pth ✅
    └── voc_bigvgan.pth ✅
``` 
3. Run `infer.sh`

`diffpitch_ts` refers to the time step of the pitch generator and `diffvoice_ts` refers to the time step of the Mel generator.

Empirically, it has been observed that if the time step of diffpitch is too small, noise remains, and if it is too large, excessive diversity occurs.

Please use it appropriately for your dataset!
```
bash infer.sh

python3 inference.py \
    --src_path './sample/src_p241_004.wav' \
    --trg_path './sample/tar_p239_022.wav' \
    --ckpt_model './ckpt/model_diffhier.pth' \
    --ckpt_voc './vocoder/voc_bigvgan.pth' \
    --output_dir './converted' \
    --diffpitch_ts 30 \
    --diffvoice_ts 6
```
🎧 Test it on your own dataset and share your interesting results! :)



## 🎓 Citation
```
@inproceedings{choi23d_interspeech,
  author={Ha-Yeong Choi and Sang-Hoon Lee and Seong-Whan Lee},
  title={{Diff-HierVC: Diffusion-based Hierarchical Voice Conversion with Robust Pitch Generation and Masked Prior for Zero-shot Speaker Adaptation}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2283--2287},
  doi={10.21437/Interspeech.2023-817}
}
```


## 💎 Acknowledgements
- Our code is based on [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC), [HiFiGAN](https://github.com/jik876/hifi-gan), and [BigVGAN](https://github.com/NVIDIA/BigVGAN).

 

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

