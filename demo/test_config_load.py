from mmcv import Config

cfg = Config.fromfile('/home/negreami/project/mmsegmentation/configs/encoder_decoder_action/cityscapes.py')
print(cfg.pretty_text)