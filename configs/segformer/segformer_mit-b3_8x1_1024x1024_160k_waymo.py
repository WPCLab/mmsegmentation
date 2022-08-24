_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_waymo.py']

model = dict(
    pretrained='/nfs/volume-807-2/darrenwang/mmseg_pretrained_models/mit_b3_pretrained.pth',
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
