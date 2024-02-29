## <div align="center">ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection</div>

### Introduction
In this paper, we propose a novel feature fusion framework of dual cross-attention transformers to model global feature interaction and capture complementary information across modalities simultaneously. In addition, we introdece an iterative interaction mechanism into dual cross-attention transformers, which shares parameters among block-wise multimodal transformers to reduce model complexity and computation cost. The proposed method is general and effective to be integrated into different detection frameworks and used with different backbones. Experimental results on KAIST, FLIR, and VEDAI datasets show that the proposed method achieves superior performance and faster inference, making it suitable for various practical scenarios. 

Paper download in: https://arxiv.org/pdf/2308.07504.pdf

### Overview
<div align="center">
  <img src="https://github.com/chanchanchan97/ICAFusion/assets/39607836/05a71809-0182-487d-9013-442497a996fd" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>

<div align="center">
  <img src="https://github.com/chanchanchan97/ICAFusion/assets/39607836/b82ba614-22da-421c-89e9-53d6d535ee36" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 2. Illustration of the proposed DMFF module </div>
</div>

### Installation
Clone repo and install requirements.txt in a Python>=3.8.0 conda environment, including PyTorch>=1.12.
```
git clone https://github.com/chanchanchan97/ICAFusion.git
cd ICAFusion
pip install -r requirements.txt
```

### Datasets
 - **KAIST**  
Link：https://pan.baidu.com/s/1UdwQJH-cHVL91pkMW-ij6g 
Code：ig3y

 - **FLIR-aligned**  
Link：https://pan.baidu.com/s/1ljr8qJYdz-60Lj-iVEHBvg 
Code：uqzs

 - **VEDAI**  
Link：https://pan.baidu.com/s/1c--8tD1s1HmV-6R2GKnACA 
Code：kp59

### Weights
 - **KAIST**  
Link：https://pan.baidu.com/s/18UXctOSgjp6EUcJXIGbWTQ
Code：9eku
 - **FLIR-aligned**
Link：https://pan.baidu.com/s/1VZbsTE4o6bw2XBypPW3zoA
Code：xli9

### Citation
If you find our work useful in your research, please consider citing:
```
@article{SHEN2023109913,
  title={ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection},
  author={Shen, Jifeng and Chen, Yifei and Liu, Yue and Zuo, Xin and Fan, Heng and Yang, Wankou},
  journal={Pattern Recognition},
  pages={109913},
  year={2023},
  issn={0031-3203},
  doi={https://doi.org/10.1016/j.patcog.2023.109913},
  author={Jifeng Shen and Yifei Chen and Yue Liu and Xin Zuo and Heng Fan and Wankou Yang},
}
```
