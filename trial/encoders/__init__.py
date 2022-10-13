

from trial.encoders.EllipseEncoder import ellipse_encoder, ellipse_mix_encoder, ellipse_self_encoder
from trial.encoders.RectEncoder import rectangle_encoder, rectangle_mix_encoder, rectangle_self_encoder
from trial.encoders.AnchorBasedEncoder import anchor_based_mix_encoder, anchor_based_self_encoder


ENCODERS = {'ellipse': ellipse_encoder,
            'ellipse-mix': ellipse_mix_encoder,
            'ellipse-self': ellipse_self_encoder,
            'rect': rectangle_encoder,
            'rect-mix': rectangle_mix_encoder,
            'rect-self': rectangle_self_encoder,
            'anchor-mix': anchor_based_mix_encoder,
            'anchor-self': anchor_based_self_encoder}


def get_encoder(encode_type):
    return ENCODERS[encode_type]
