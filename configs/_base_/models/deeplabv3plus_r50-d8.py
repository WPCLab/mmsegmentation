# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[
                2.831637118444236, 5.056285632204348, 5.359396933797303, 7.141446981306911,
                7.83548537241383, 7.369427346768545, 5.159373743076446, 4.76376838541227,
                5.74627485203493, 4.255490074972682, 8.47767089648915, 6.7950642130560635,
                6.853404414400033, 1.551886637845778, 1.358233603952349, 4.278350321743853,
                1.137561790141584, 4.282902057952668, 3.7958329552557286, 5.35584437600218,
                3.001029493800046
            ])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[
                2.831637118444236, 5.056285632204348, 5.359396933797303, 7.141446981306911,
                7.83548537241383, 7.369427346768545, 5.159373743076446, 4.76376838541227,
                5.74627485203493, 4.255490074972682, 8.47767089648915, 6.7950642130560635,
                6.853404414400033, 1.551886637845778, 1.358233603952349, 4.278350321743853,
                1.137561790141584, 4.282902057952668, 3.7958329552557286, 5.35584437600218,
                3.001029493800046
            ])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
