_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_ohem.py',
    '../_base_/datasets/mapillary_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
