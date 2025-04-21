# TFS-Diff: 三模态医学图像融合与超分辨率扩散模型

## 简介

TFS-Diff是一个基于条件扩散模型的三模态医学图像融合与超分辨率框架。该模型通过集成TriModalRestormer精细化网络，能够同时处理三种不同模态的医学图像（如可见光、红外和热成像），并生成高质量的融合超分辨率结果。

## 主要特点

- **三模态融合**：同时处理三种不同模态的医学图像输入
- **超分辨率**：将低分辨率图像提升到高分辨率
- **精细化网络**：使用TriModalRestormer网络对扩散模型的输出进行精细化处理
- **高质量结果**：生成细节丰富、对比度高的医学图像

## 模型架构

该模型基于Diff-IF架构，并添加了以下改进：

1. **三模态输入处理**：能够同时处理三种不同模态的医学图像
2. **TriModalRestormer精细化网络**：在扩散过程的最后几个步骤中应用，提升图像质量
3. **改进的采样过程**：在p_sample和p_mean_variance函数中集成了精细化网络

## 使用方法

### 训练

使用以下命令开始训练：

```bash
python sr.py -c config/tfs_diff_config.json -p train
```

### 推理

使用以下命令进行推理：

```bash
python infer.py -c config/tfs_diff_config.json -p val -path [checkpoint_path]
```

## 配置文件说明

配置文件`tfs_diff_config.json`中的关键参数：

```json
"model": {
    "which_model_G": "tfs",  // 使用TFS-Diff模型
    "use_trimodal_refiner": true,  // 启用三模态精细化网络
    "trimodal_refiner_config": {  // 精细化网络配置
        "in_channel": 3,
        "out_channel": 3,
        "dim": 36,
        "num_blocks": [2, 2],
        "num_refinement_blocks": 2,
        "heads": [1, 2, 4, 8],
        "ffn_expansion_factor": 2.66,
        "bias": false,
        "LayerNorm_type": "WithBias"
    }
}
```

## 数据集准备

数据集应包含以下目录结构：

```
data/
  train/
    1_lr_128/      # 模态1低分辨率
    1_sr_128_256/  # 模态1超分辨率
    2_lr_128/      # 模态2低分辨率
    2_sr_128_256/  # 模态2超分辨率
    3_lr_128/      # 模态3低分辨率
    3_sr_128_256/  # 模态3超分辨率
    hr_256/        # 高分辨率目标
  val/
    ...(同上)
```

## 注意事项

- 训练时需要确保三个模态的图像对齐
- 精细化网络仅在扩散过程的最后几个步骤中应用，以平衡计算效率和图像质量
- 对于不同的医学图像模态，可能需要调整精细化网络的参数

## 引用

如果您使用了本模型，请引用以下论文：

```
@article{TFS-Diff2024,
  title={Simultaneous Tri-Modal Medical Image Fusion and Super-Resolution using Conditional Diffusion Model},
  author={},
  journal={},
  year={2024}
}
```