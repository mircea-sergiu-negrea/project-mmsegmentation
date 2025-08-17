from mmcv import Config

cfg = Config.fromfile('/home/negreami/project/mmsegmentation/configs/encoder_decoder_action/ap4ad_rgb.py')
print(cfg.pretty_text)