
import torch
import torch.nn as nn

from thanh_transformers.x_transformers import TransformerWrapper, Decoder
from thanh.models.vision_transformer_hybrid import HybridEmbed
from thanh.models.resnetv2 import ResNetV2
from thanh.models.layers import StdConv2dSame

from CustomVisionTransformer import CustomVisionTransformer
from CustomARWrapper import CustomARWrapper


class Model(nn.Module):
    def __init__(self, encoder: CustomVisionTransformer, decoder: CustomARWrapper, args, temp: float = .333):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_token = args.bos_token
        self.eos_token = args.eos_token
        self.max_seq_len = args.max_seq_len
        self.temperature = temp

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        device = x.device
        encoded = self.encoder(x.to(device))
        dec = self.decoder.generate(torch.LongTensor([self.bos_token]*len(x))[:, None].to(device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec


def get_model(args, training=False):
    backbone = ResNetV2(
        layers=args.backbone_layers, num_classes=0, global_pool='', in_chans=args.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    min_patch_size = 2**(len(args.backbone_layers)+1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps//min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(img_size=(args.max_height, args.max_width),
                                      patch_size=args.patch_size,
                                      in_chans=args.channels,
                                      num_classes=0,
                                      embed_dim=args.dim,
                                      depth=args.encoder_depth,
                                      num_heads=args.heads,
                                      embed_layer=embed_layer
                                      ).to(args.device)

    decoder = CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token
    ).to(args.device)
    if 'wandb' in args and args.wandb:
        import wandb
        wandb.watch((encoder, decoder.net.attn_layers))
    model = Model(encoder, decoder, args)
    if training:
        # check if largest batch can be handled by system
        im = torch.empty(args.batchsize, args.channels, args.max_height,
                         args.min_height, device=args.device).float()
        seq = torch.randint(0, args.num_tokens, (args.batchsize,
                            args.max_seq_len), device=args.device).long()
        decoder(seq, context=encoder(im)).sum().backward()
        model.zero_grad()
        torch.cuda.empty_cache()
        del im, seq
    return model
