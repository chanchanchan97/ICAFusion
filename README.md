## <div align="center">ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection</div>

### Introduction
Effective feature fusion of multispectral images plays a crucial role in multispectral object detection. Previous studies have demonstrated the effectiveness of feature fusion using convolutional neural networks, but these methods are sensitive to image misalignment due to the inherent deficiency in local-range feature interaction resulting in the performance degradation. To address this issue, a novel feature fusion framework of dual cross-attention transformers is proposed to model global feature interaction and capture complementary information across modalities simultaneously. This framework enhances the discriminability of object features through the query-guided cross-attention mechanism, leading to improved performance. However, stacking multiple transformer blocks for feature enhancement incurs a large number of parameters and high spatial complexity. To handle this, inspired by the human process of reviewing knowledge, an iterative interaction mechanism is proposed to share parameters among block-wise multimodal transformers, reducing model complexity and computation cost. The proposed method is general and effective to be integrated into different detection frameworks and used with different backbones. Experimental results on KAIST, FLIR, and VEDAI datasets show that the proposed method achieves superior performance and faster inference, making it suitable for various practical scenarios.

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
