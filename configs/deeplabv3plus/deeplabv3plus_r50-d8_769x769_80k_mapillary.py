_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/mapillary_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(align_corners=True, num_classes=66),
    auxiliary_head=dict(align_corners=True, num_classes=66),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
