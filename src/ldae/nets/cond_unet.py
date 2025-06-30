
import torch
import torch.nn as nn

from src.ldae.modules.module import linear, conv_nd, normalization, zero_module, timestep_embedding
from src.ldae.modules.module import ResBlock, ResBlockShift, AttentionBlock, TimestepSequential

"""
Originally ported from here: https://github.com/ckczzj/PDAE/tree/master and adapted for the LDAE framework.
"""


class CondUNet(nn.Module):
    """
    Conditional UNet in which we use shift-and scale AdapGN at every level of UNet. 
    Unlike LDAE code: in this setting the CondUnet is trained from the start (instead of pretrain unconditionally first).
    :param input_channel: channels in the input Tensor.
    :param base_channel: base channel count for the model.
    :param num_residual_blocks_of_a_block: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_multiplier: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param latent_dim: latent dim
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param use_scale_shift_norm: default True, use FiLM style conditioning mechanism. If False, then condition is added to hidden state.
                                h = h + cond_emb. 
    """

    def __init__(
            self,
            input_channel,
            base_channel,
            channel_multiplier,
            num_residual_blocks_of_a_block,
            attention_resolutions,
            num_heads,
            head_channel,
            use_new_attention_order,
            dropout,
            latent_dim,
            dims=2,
            learn_sigma=False,
            use_scale_shift_norm=True, 
            **kwargs
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.base_channel = base_channel
        output_channel = input_channel * 2 if learn_sigma else input_channel

        time_embed_dim = base_channel * 4

        self.time_embed = nn.Sequential(
            linear(base_channel, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # original class label
        # self.label_emb = nn.Embedding(latent_dim, time_embed_dim)

        # free representation learning
        # this layer is trainable
        self.label_emb = nn.Linear(latent_dim, time_embed_dim)

        ch = input_ch = int(channel_multiplier[0] * base_channel)
        self.input_blocks = nn.ModuleList(
            [TimestepSequential(conv_nd(dims, input_channel, ch, 3, padding=1))]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        # shift_input_block_chans = [ch]
        ds = 1
        # shift_ds = 1
        for level, mult in enumerate(channel_multiplier):
            for _ in range(num_residual_blocks_of_a_block):
                layers = [
                    ResBlockShift(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * base_channel),
                        dims=dims,
                        use_scale_shift_norm=self.use_scale_shift_norm
                    )
                ]
                ch = int(mult * base_channel)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                self.input_blocks.append(TimestepSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                # shift_input_block_chans.append(ch)
            if level != len(channel_multiplier) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepSequential(
                        ResBlockShift(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True,
                            use_scale_shift_norm=self.use_scale_shift_norm
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                # shift_input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
                # shift_ds *= 2

        self.middle_block = TimestepSequential(
            ResBlockShift(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=head_channel,
                use_new_attention_order=use_new_attention_order
            ),
            ResBlockShift(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
        )
        self._feature_size += ch

        memory_ch = ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_multiplier))[::-1]:
            for i in range(num_residual_blocks_of_a_block + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockShift(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channel * mult),
                        dims=dims,
                        use_scale_shift_norm=self.use_scale_shift_norm
                    )
                ]
                ch = int(base_channel * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                if level and i == num_residual_blocks_of_a_block:
                    out_ch = ch
                    layers.append(
                        ResBlockShift(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True,
                            use_scale_shift_norm=self.use_scale_shift_norm
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepSequential(*layers))
                self._feature_size += ch

        # ch = memory_ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, output_channel, 3, padding=1)),
        )

    def forward(self, x, time, condition):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param time: a 1-D batch of timesteps.
        :param condition: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(time, self.base_channel))
        shift_emb = self.label_emb(condition)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, shift_emb)
            hs.append(h)
        h = self.middle_block(h, emb, shift_emb)
        for module in self.output_blocks:
            h_prev = hs.pop()
            h = torch.cat([h, h_prev], dim=1)
            h = module(h, emb, shift_emb)
        return self.out(h)