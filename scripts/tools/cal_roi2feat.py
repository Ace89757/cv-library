import torch


def cal_rot2feat_levels(rois, num_levels, finest_scale):
    scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()

    print(target_lvls)



if __name__ == '__main__':
    # 这里的rois是在input_size下的尺寸

    # 期望:
    # 8 -> 0
    # 64 -> 1
    # 256 -> 2
    # 512 -> 3
    rois = torch.tensor(
        [
            [0, 46.3, 46.3, 54.3, 54.3],
            [1, 46.3, 46.3, 110.3, 110.3],
            [2, 46.3, 46.3, 310.3, 310.3],
            [3, 46.3, 46.3, 558.3, 558.3],

        ]
    )
    print(rois.shape)

    cal_rot2feat_levels(rois, 4, 24)