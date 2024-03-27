## AnchorGenerator使用说明

### 文件位置: 
    mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py


**finest_scale:**

    通过该参数可控制将不同的rois分配到不同的特征层提取特征，默认=56

    分配规则：
        scale < finest_scale * 2 = 112的目标都会分配到x[0]=P2上提取特征
        scale in [finest_scale * 2, finest_scale * 4] = [112, 224) 的目标都会分配到x[1]=P3上提取特征
        scale in [finest_scale * 4, finest_scale * 8] = [224, 448) 的目标都会分配到x[2]=P4上提取特征
        scale in [finest_scale * 8, inf] = [448, inf) 的目标都会分配到x[2]=P5上提取特征