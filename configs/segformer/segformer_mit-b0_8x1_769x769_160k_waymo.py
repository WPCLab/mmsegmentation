_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/waymo_769x769.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='/nfs/volume-807-2/darrenwang/mmseg_pretrained_models/mit_b0_pretrained.pth',
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', min_kept=100000),
        align_corners=True,
        num_classes=66))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=1, workers_per_gpu=1)

checkpoint_config = dict(interval=2000)
