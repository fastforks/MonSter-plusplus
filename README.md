# MonSter-plusplus: faster inference fork

> **You are on `cuda-kernels`.** Research fork of [Junda24/MonSter-plusplus](https://github.com/Junda24/MonSter-plusplus) (base `9167101`; `main` mirrors upstream). Custom CUDA kernels, so this branch does not export to ONNX / TensorRT; there is no `portable` branch for this repo.
>
> What needs the kernels: the fp16 fast path in the `*_fast.py` / `inference.py` files (cuDNN-frontend fp16-accumulate convolutions, CUTLASS fused-epilogue convs, cost-volume / correlation kernels). Upstream's own modules and weights are unchanged.
>
> Build: prebuilt cubins (sm_80/86/89/90/120) and one CUTLASS shared library sit in `prebuilt/` next to the code (each < 3 MB). If a prebuilt does not match your GPU or the source hash, the code compiles it with `nvcc` from `CUDA_HOME` (default `/usr/local/cuda`) into `~/.cache`; the CUTLASS library is only rebuilt from source with `CUTLASS_DIR` set (CUTLASS v4.7.1 headers). fp16-accumulate convs need `pip install nvidia-cudnn-frontend`.
>

> Extra dependency: [`fastbb`](https://github.com/fastforks/fastbb) (fast backbones), `pip install` it or put it on `PYTHONPATH`.
>

> Licence, as upstream (MonSter-plusplus Model License v1.0, see LICENSE)


Released weights, RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-23. KITTI = 000000_10 (1242x375), Midd-H = Middlebury trainingH Motorcycle (1482x994). Median of 10 runs after warm-up (the first-call column is the very first call of a fresh process). Peak memory = torch peak allocated / reserved over the whole run (both sizes). EPE = mean over 3 ETH3D + 3 Middlebury-H training pairs (short check, not a benchmark; a pair that ran out of memory in any row is left out of the EPE mean of all rows). flags = set from the driver only, no upstream source edit: `torch.inference_mode()`, `cudnn.benchmark=True`, TF32 allowed, plus the fastest combination that keeps EPE within noise (listed in brackets in the row). `torch.compile` is not part of the row.

MonSter++

| version | KITTI ms | Midd-H ms | speed-up at H (vs flags / vs upstream) | peak MB (alloc / reserved) | startup / first call / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|
| upstream (fp32) | 1005.6 | 4335.8 | 1.00x | 6388 / 7638 | 8.1 s / 1.22 s / 4.34 s | 0.107 | 0.928 |
| upstream + flags (fp16 autocast, cudnn.benchmark, TF32, inference_mode) | 673.2 | 3194.3 | 1.00x / 1.36x | 10636 / 12490 | 6.9 s / 4.93 s / 10.86 s | 0.106 | 0.934 |
| **cuda-kernels** | 189.2 | 676.2 | 4.72x / 6.41x | 2438 / 2598 | 7.3 s / 0.57 s / 0.74 s | 0.106 | 0.933 |

Max / mean abs disparity difference cuda-kernels vs upstream: 120 px / 0.043 px (KITTI + 6 EPE pairs; the maximum is set by a few single pixels).

RT-MonSter++

| version | KITTI ms | Midd-H ms | speed-up at H (vs flags / vs upstream) | peak MB (alloc / reserved) | startup / first call / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|
| upstream (fp32) | 91.2 | 251.1 | 1.00x | 2927 / 3556 | 3.2 s / 0.64 s / 0.25 s | 0.146 | 5.688 |
| upstream + flags (fp16 autocast, cudnn.benchmark, TF32, inference_mode) | 67.4 | 176.1 | 1.00x / 1.43x | 6271 / 6364 | 2.9 s / 3.57 s / 3.30 s | 0.146 | 5.690 |
| **cuda-kernels** | 42.2 | 119.5 | 1.47x / 2.10x | 2613 / 2674 | 3.1 s / 0.71 s / 0.15 s | 0.147 | 5.685 |

Max / mean abs disparity difference cuda-kernels vs upstream: 99.7 px / 0.0549 px (KITTI + 6 EPE pairs; the maximum is set by a few single pixels).


---

# 🚀 MonSter++ 🚀 
MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors  <a href="https://arxiv.org/abs/2501.08643"><img src="https://img.shields.io/badge/arXiv-2501.08643-b31b1b?logo=arxiv" alt='arxiv'></a>

<p align="center">
</p>
<h2>🤗 Demo Video：</h2>
<div align="center">
	<a href="https://www.youtube.com/watch?v=YIhQCuQAKrU">
		<img src="Media/demo.png" width="50%" alt="Overview Video">
	</a>
</div>
<p></p>

<div align="center">

</div>

# 🚀 RT-MonSter++ 🚀 
- `[2025/12]` We release the ONNX and TensorRT export and conversion scripts for RT-MonSter++, enabling efficient inference on mobile and edge devices. Further optimizations are planned, including integrating techniques from DepthAnything to further improve efficiency!
- <img width="300" height="220" alt="45a82bb73e85fb4a66ade8bd3c441b48" src="https://github.com/user-attachments/assets/c34a5e24-8327-4b61-9e17-66d286e2be2c" />

## News 
- `[2025/12]` The most often asked about the issue: ONNX and tensorRT export problems: can be reference to https://github.com/Junda24/MonSter-plusplus/pull/4
- `[2025/9]` We have open-sourced our lightweight real-time model RT-MonSter++
- `[2025/9]` Weights for RT-MonSter++ model released!
- `[2025/10]` We have open-sourced our accuracy model MonSter++
- `[2025/10]` Weights for MonSter++ model released!

## ✈️ MonSter++ Model weights (accuracy models)

| Model      |                                               Link                                                |
|:----:|:-------------------------------------------------------------------------------------------------:|
| KITTI| [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/KITTI_large.pth?download=true) |
| ETH3D | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/ETH3D_large.pth?download=true)|
|Middlebury | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/Middlebury_large.pth?download=true)|
|mix_all | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/Mix_all_large.pth?download=true)|

The mix_all model is trained on all the datasets we collect over 2M image pairs, which has the best performance on zero-shot generalization.

## ✈️ RT-MonSter++ Model weights (light weight models)

| Model      |                                               Link                                                |
|:----:|:-------------------------------------------------------------------------------------------------:|
| KITTI 2012| [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/KITTI_2012.pth?download=true) |
| KITTI 2015 | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/KITTI_2015.pth?download=true)|
|mix_all | [Download 🤗](https://huggingface.co/cjd24/MonSter-plusplus/resolve/main/Zero_shot.pth?download=true)|

The mix_all model is trained on all the datasets we collect over 2M image pairs, which has the best performance on zero-shot generalization.

### 🎬 Dependencies

```Shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.6.13
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install openexr
pip install pyexr
pip install imath
pip install h5py
pip install swanlab

```

# Leaderboards 🏆
We obtained the 1st place on the world-wide [KITTI 2012 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI 2015 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

1. [KITTI 2012 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
<img width="1221" height="696" alt="image" src="https://github.com/user-attachments/assets/886445d2-c9c2-4148-9bd2-599e62802e96" />

2. [KITTI 2015 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

<img width="1205" height="709" alt="image" src="https://github.com/user-attachments/assets/161d5344-0a10-4e93-9aa4-ea99f9bfb349" />

 We obtained the 2nd place on the world-wide [ETH3D leaderboard](https://www.eth3d.net/low_res_two_view), while maintaining the lowest inference cost, particularly compared with the top-ranked method.

 3. [ETH3D leaderboard](https://www.eth3d.net/low_res_two_view)
 <img width="2543" height="1363" alt="3e06f3c5a624ab19c78fb89c0f516ed2" src="https://github.com/user-attachments/assets/8168ee8c-6003-4ca8-8efe-b85030139017" />


## ✈️ Citation

If you find our works useful in your research, please consider citing our papers:

```bibtex

MonSter：
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Junda and Liu, Longliang and Xu, Gangwei and Wang, Xianqi and Zhang, Zhaoxing and Deng, Yong and Zang, Jinliang and Chen, Yurui and Cai, Zhipeng and Yang, Xin},
    title     = {MonSter: Marry Monodepth to Stereo Unleashes Power},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6273-6282}
}

MonSter++：
@misc{cheng2025monsterunifiedstereomatching,
      title={MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors}, 
      author={Junda Cheng and Wenjing Liao and Zhipeng Cai and Longliang Liu and Gangwei Xu and Xianqi Wang and Yuzhou Wang and Zikang Yuan and Yong Deng and Jinliang Zang and Yangyang Shi and Jinhui Tang and Xin Yang},
      year={2025},
      eprint={2501.08643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.08643}, 
}
```
