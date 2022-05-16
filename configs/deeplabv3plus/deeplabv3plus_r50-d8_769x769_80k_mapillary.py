_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/mapillary_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', min_kept=100000),
        align_corners=True,
        num_classes=66),
    auxiliary_head=dict(
        sampler=dict(type='OHEMPixelSampler', min_kept=100000),
        align_corners=True,
        num_classes=66))
