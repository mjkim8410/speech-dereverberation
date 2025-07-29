import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# 1. Normalization Layers
###############################################################################
class GlobalChannelLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization for 1D data (B, C, T).
    Normalizes across T dimension, separately for each channel.
    """
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, T)
        # Reshape to (B*C, T) to do layernorm over T for each channel separately
        B, C, T = x.shape
        x_reshaped = x.view(B * C, -1)
        
        mean = x_reshaped.mean(dim=1, keepdim=True)
        var = x_reshaped.var(dim=1, keepdim=True, unbiased=False)
        
        x_norm = (x_reshaped - mean) / (torch.sqrt(var + self.eps))
        x_norm = x_norm.view(B, C, T)
        
        return self.gamma.view(1, C, 1) * x_norm + self.beta.view(1, C, 1)


###############################################################################
# 2. TCN Building Blocks
###############################################################################
class TCNBlock(nn.Module):
    """
    One temporal convolution block. Uses:
      - 1x1 Conv for bottleneck
      - Depthwise Separable Conv (or plain Conv1d) with dilation
      - GLN (channel-wise LN)
      - Residual + Skip Connection
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation,
        padding,
        causal=False
    ):
        super().__init__()
        self.causal = causal

        # 1x1 bottleneck
        self.bottleneck = nn.Conv1d(in_channels, hidden_channels, 1)

        # depthwise conv
        self.depthwise = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            groups=hidden_channels,
            dilation=dilation,
            padding=padding
        )
        self.prelu = nn.PReLU()
        self.norm = GlobalChannelLayerNorm(hidden_channels)

        # 1x1 pointwise after depthwise
        self.pointwise = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x):
        """
        x shape: (B, in_channels, T)
        return shape: (B, in_channels, T)
        """
        residual = x

        # Bottleneck
        x = self.bottleneck(x)
        # Depthwise
        x = self.depthwise(x)
        
        # If causal, we might slice off the extra frames introduced by padding
        # This is optional, depends on how you do padding. Example:
        # if self.causal:
        #     x = x[:, :, :-self.padding_amount]  # or something similar

        # Nonlinearity + Norm
        x = self.prelu(x)
        x = self.norm(x)

        # Pointwise
        x = self.pointwise(x)

        # Residual + skip
        return x + residual


class TCNStack(nn.Module):
    """
    A stack of TCN blocks with increasing dilation. We repeat this stack.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        layers_per_stack,
        dilation_base=2,
        causal=False
    ):
        super().__init__()
        blocks = []
        for i in range(layers_per_stack):
            # Dilation grows exponentially for each layer
            dilation = dilation_base ** i
            # Padding to maintain same output length
            # If causal==False, we use "same" style padding
            # If kernel_size=3, dilation=4 -> padding=4
            # => so output length matches input length
            padding = (kernel_size - 1) * dilation // (2 if not causal else 1)
            blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    causal=causal
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)


###############################################################################
# 3. Full Conv-TasNet
###############################################################################
class ConvTasNet(nn.Module):
    """
    Full-fledged Conv-TasNet model with:
      - Encoder (learned filterbank)
      - Multiple repeated TCN stacks
      - Mask generation for multiple sources
      - Decoders (one per source) or a single decoder + mask slicing

    Default parameters reflect typical values from the literature.
    """
    def __init__(
        self,
        num_sources=1,
        encoder_kernel_size=16,  # L
        encoder_stride=8,       # hop
        encoder_filters=1024,    # N
        # TCN parameters
        tcn_hidden=128,         # B
        tcn_kernel_size=5,
        tcn_layers=8,           # X
        tcn_stacks=4,           # R (how many repeats of the TCN stack)
        causal=False
    ):
        super().__init__()
        
        self.num_sources = num_sources
        self.encoder_filters = encoder_filters

        # -----------------
        # 3.1 Encoder
        # -----------------
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_filters,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            padding=encoder_kernel_size // 2,
            bias=False
        )
        # Encoded shape => (B, N, frames)

        # -----------------
        # 3.2 TCN Separator
        # -----------------
        # We do an initial 1x1 conv "bottleneck" after the encoder
        self.bottleneck = nn.Conv1d(encoder_filters, tcn_hidden, 1)

        # Build repeated stacks of TCN
        tcn_blocks = []
        for _ in range(tcn_stacks):
            tcn_blocks.append(
                TCNStack(
                    in_channels=tcn_hidden,
                    hidden_channels=tcn_hidden,
                    kernel_size=tcn_kernel_size,
                    layers_per_stack=tcn_layers,
                    causal=causal
                )
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        
        # Final 1x1 conv to map back to (num_sources * encoder_filters)
        self.mask_conv = nn.Conv1d(tcn_hidden, num_sources * encoder_filters, 1)

        # -----------------
        # 3.3 Decoder
        # -----------------
        # We'll use a single decoder, but we produce multiple masked features,
        # then decode each one. The decoder "inverse" of the encoder is typically
        # a transpose convolution or overlap-add approach.
        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_filters,
            out_channels=1,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            padding=encoder_kernel_size // 2,
            bias=False
        )

    def forward(self, mixture):
        """
        mixture: (B, T)  -- single-channel mixture
        returns: (B, num_sources, T')  separated waveforms
        """
        # print(mixture.shape)
        mixture = mixture.squeeze(1)
        B, T = mixture.shape
        # Add channel dim
        mixture = mixture.unsqueeze(1)  # (B, 1, T)

        # 1) Encoder -> (B, N, frames)
        mixture_w = self.encoder(mixture)

        # 2) Bottleneck -> TCN -> mask predictions
        x = self.bottleneck(mixture_w)  # (B, hidden, frames)
        x = self.tcn(x)                 # (B, hidden, frames)

        # Produce (num_sources * N) channels
        mask_out = self.mask_conv(x)    # (B, num_sources*N, frames)

        # Reshape to (B, num_sources, N, frames)
        B2, CN, F = mask_out.shape
        mask_out = mask_out.view(B2, self.num_sources, self.encoder_filters, F)

        # 3) Apply mask to mixture_w
        # mixture_w: (B, N, frames)
        source_waves = []
        for i in range(self.num_sources):
            mask_i = mask_out[:, i, :, :]  # (B, N, frames)
            mask_i = torch.sigmoid(mask_i) # or ReLU / softmax, depends on preference
            masked_features = mixture_w * mask_i  # (B, N, frames)

            # 4) Decoder for each source
            wave_i = self.decoder(masked_features)
            # wave_i shape => (B, 1, time')
            source_waves.append(wave_i)

        # Stack them => (B, num_sources, time')
        out = torch.cat(source_waves, dim=1)
        return out

