# Add support for jsonargparse >= 4.38.0
# jsonargparse>=4.38.0 enforces tighter nested schema validation compared to older versions
# we need to define dataclass to avoid error: "nested key doesn't exist" when running validate

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class UnetArgs:
    input_channel: int = 1
    base_channel: int = 64
    channel_multiplier: List[int] = field(default_factory=lambda: [1, 2, 3])
    num_residual_blocks_of_a_block: int = 1
    attention_resolutions: List[int] = field(default_factory=lambda: [2, 4])
    num_heads: int = 1
    head_channel: int = -1
    use_new_attention_order: bool = False
    dropout: float = 0.1
    dims: int = 2


@dataclass
class BackboneArgs:
    net_class_path: str = None
    weights: Optional[str] = None
    freeze_perc: float = 0.0
    grayscale: bool = True

@dataclass
class EncArgs:
    backbone_args: BackboneArgs
    emb_chans: int = 512
    seq_len: int = 128

@dataclass
class timesteps_args:
    timesteps: int = 1000
    betas_type: str = "linear"

