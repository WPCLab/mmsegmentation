_base_ = ['./segformer_mit-b0_8x1_769x769_160k_mapillary.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b4.pth'),
        embed_dims=64,
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
