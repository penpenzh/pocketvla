"""PocketVLA model (HuggingFace Transformers format; presets: 3m / 50m / 150m).

Early-fusion VLM: CNN vision encoder -> 64 grid tokens; token/pos embeddings for
text; proprio projection -> 1 state token. The trunk runs [vision | text | proprio
| action-query] through a Pre-LN bidirectional Transformer with a padding mask.
The action head reads the action-query hidden state; aux spatial readouts (ball/
bin world coordinates from fused vision tokens) are deep supervision only, and
disappear with use_aux_heads=False.

Save/load: model.save_hf(dir, tokenizer) / PocketVLA.load_hf(dir, device), or
transformers.AutoModel/AutoTokenizer.from_pretrained(dir, trust_remote_code=True).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel, PreTrainedTokenizer)

EE_NORM = 1.1   # normalization scale for ee_x/ee_y in proprio (matches sim/arm2d._obs)


# Size presets: CNN channels / stride-1 blocks / trunk (d_model, layers, FFN)
# / action MLP widths / aux readout hidden width
MODEL_PRESETS = {
    "3m":   dict(ch=(24, 48, 72, 96, 96),       extra=4,
                 d_model=160, n_layer=3, d_ff=1024,
                 act=(1280, 640), ground_hidden=48),
    "50m":  dict(ch=(80, 160, 240, 320, 320),   extra=7,
                 d_model=384, n_layer=6, d_ff=6144,
                 act=(4096, 2048), ground_hidden=96),
    "150m": dict(ch=(96, 192, 320, 448, 448),   extra=12,
                 d_model=768, n_layer=7, d_ff=7168,
                 act=(8192, 4096), ground_hidden=128),
}

# kwargs that override the preset architecture (mirror config.ARCH_KEYS;
# duplicated to avoid an import cycle)
ARCH_OVERRIDE_KEYS = ("ch", "extra_blocks", "d_model", "n_layer", "d_ff",
                      "act_hidden", "ground_hidden", "use_aux_heads")


def _gn(ch: int) -> int:
    return 4 if ch == 16 else 8


class PocketVLAConfig(PretrainedConfig):
    """VLA model config (HF format, JSON-serializable / rebuildable from config.json)."""

    model_type = "pocketvla"

    def __init__(self, size: str = "3m", obs_res: int = 64,
                 world_extent: float = 1.25, proprio_dim: int = 6,
                 action_dim: int = 3, lang_len: int = 10, vocab_size: int = 64,
                 action_max: float = 0.15, ground_hidden: int = 64,
                 use_aux_heads: bool = True, **kwargs):
        self.size = size
        # unknown size labels fall back to the 3m base; yaml model: overrides apply below
        p = MODEL_PRESETS.get(size, MODEL_PRESETS["3m"])
        self.obs_res = obs_res
        self.world_extent = world_extent
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.lang_len = lang_len
        self.vocab_size = vocab_size
        self.action_max = action_max
        self.ch = list(p["ch"])
        self.extra_blocks = p["extra"]
        self.d_model = p["d_model"]
        self.n_layer = p["n_layer"]
        self.d_ff = p["d_ff"]
        self.act_hidden = list(p["act"])
        self.ground_hidden = p["ground_hidden"]
        # aux readout heads on/off (False = pure action head, ablation mode)
        self.use_aux_heads = bool(use_aux_heads)
        # architecture overrides from the yaml `model:` section (kwargs)
        for k in ARCH_OVERRIDE_KEYS:
            if k in kwargs:
                v = kwargs.pop(k)
                setattr(self, k, list(v) if isinstance(v, (list, tuple)) else v)
        super().__init__(**kwargs)


class MaskedMHSA(nn.Module):
    """Multi-head self-attention with key padding mask (plain matmul + softmax)."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.dh = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B, h, L, dh)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)   # (B, h, L, L)
        att = att.masked_fill(~pad[:, None, None, :], float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHSA(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    """CNN visual tokenizer: 64x64x3 -> 8x8xC spatial feature map (= 64 grid tokens)."""

    def __init__(self, obs_res: int = 64, channels=(16, 32, 48, 64, 64),
                 extra_blocks: int = 0):
        super().__init__()
        layers = []
        in_ch = 3
        for i, ch in enumerate(channels):
            stride = 2 if i < 3 else 1
            layers += [nn.Conv2d(in_ch, ch, 3, stride, 1),
                       nn.GroupNorm(_gn(ch), ch), nn.SiLU()]
            in_ch = ch
        for _ in range(extra_blocks):
            layers += [nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                       nn.GroupNorm(_gn(in_ch), in_ch), nn.SiLU()]
        self.features = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)                   # (B, C, 8, 8)


class MultimodalLM(nn.Module):
    """Fused trunk: [vision | text | extra] tokens -> Pre-LN bidirectional
    Transformer (token-level cross-modal attention, padding mask)."""

    def __init__(self, vocab_size: int, d_model: int, n_head: int,
                 n_layer: int, d_ff: int, lang_len: int, n_vis: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos_text = nn.Embedding(lang_len, d_model)
        self.pos_vis = nn.Embedding(n_vis, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff) for _ in range(n_layer)]
        )
        self.n_vis = n_vis

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor,
                vis: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """Returns hidden states (B, n_vis+L+K, d_model), same token order."""
        B, L = tokens.shape
        K = extra.shape[1]
        pos_t = torch.arange(L, device=tokens.device)
        pos_v = torch.arange(self.n_vis, device=tokens.device)
        h = torch.cat([
            vis + self.pos_vis(pos_v),            # vision tokens (spatial pos emb)
            self.tok(tokens) + self.pos_text(pos_t),   # text tokens
            extra,                                # proprio + action query
        ], dim=1)
        ones_v = pad.new_ones((B, self.n_vis))
        ones_k = pad.new_ones((B, K))
        pad_full = torch.cat([ones_v, pad, ones_k], dim=1)
        for blk in self.blocks:
            h = blk(h, pad_full)
        return h


class SpatialReadout(nn.Module):
    """Attention readout of one world coordinate from the fused token grid.

    Language is already fused into the trunk states (the ball head simply has to
    localize the language-relevant cell, the bin head the bin's appearance),
    so no explicit language conditioning (FiLM) is needed here.
    """

    def __init__(self, d_model: int, hidden: int, grid: int,
                 cell_centers: torch.Tensor):
        super().__init__()
        self.grid = grid
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, hidden, 3, padding=1),
            nn.GroupNorm(_gn(hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.register_buffer("centers", cell_centers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: (B, n_vis, d_model) fused vision-token hidden states."""
        B, n_vis, D = states.shape
        fmap = states.transpose(1, 2).reshape(B, D, self.grid, self.grid)
        logits = self.conv(fmap).flatten(1)
        attn = logits.softmax(dim=-1)
        return attn @ self.centers


def _cell_centers(obs_res: int, world_extent: float) -> torch.Tensor:
    """World coordinates of the 8x8 grid cell centers."""
    n = obs_res // 8
    idx = np.arange(n, dtype=np.float32)
    px = idx * (obs_res / n) + (obs_res / n / 2 - 0.5)
    xs = (px + 0.5) / obs_res * 2 * world_extent - world_extent
    ys = world_extent - (px + 0.5) / obs_res * 2 * world_extent
    return torch.from_numpy(
        np.array([[xs[j], ys[i]] for i in range(n) for j in range(n)],
                 dtype=np.float32))


class PocketVLA(PreTrainedModel):
    """VLA model (PreTrainedModel subclass, from_pretrained/save_pretrained compatible)."""

    config_class = PocketVLAConfig
    base_model_prefix = "pocketvla"

    def __init__(self, config: PocketVLAConfig):
        super().__init__(config)
        c = self.config
        # Per-dim action scaling: joint deltas x action_max, grip dim x 1 (keeps tanh's +/-1 semantics)
        self.register_buffer("action_scale",
                             torch.tensor([c.action_max, c.action_max, 1.0]))
        n_head = max(4, min(12, c.d_model // 64))
        assert c.d_model % n_head == 0
        grid = c.obs_res // 8
        n_vis = grid * grid

        # Visual tokenizer -> LM embedding space
        self.vision = VisionEncoder(c.obs_res, c.ch, c.extra_blocks)
        self.vis_proj = nn.Linear(c.ch[-1], c.d_model)
        # Fused vision-language trunk (LM)
        self.lm = MultimodalLM(c.vocab_size, c.d_model, n_head,
                               c.n_layer, c.d_ff, c.lang_len, n_vis)
        # proprio token + learnable action-query token (an nn.Embedding so
        # _init_weights covers it reproducibly)
        self.proprio_proj = nn.Linear(c.proprio_dim, c.d_model)
        self.action_query = nn.Embedding(1, c.d_model)
        # aux spatial readouts (deep supervision; never consumed by the action head)
        if getattr(c, "use_aux_heads", True):
            centers = _cell_centers(c.obs_res, c.world_extent)
            self.grounding = SpatialReadout(c.d_model, c.ground_hidden, grid, centers.clone())
            self.bin_grounding = SpatialReadout(c.d_model, c.ground_hidden, grid, centers.clone())
        # Action head: reads the LM output at the action-query token
        h1, h2 = c.act_hidden
        self.action_head = nn.Sequential(
            nn.Linear(c.d_model, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, c.action_dim), nn.Tanh(),
        )
        self.post_init()

    def _init_weights(self, module):
        # nn.Embedding covers tok/pos embeddings AND the action query (a bare
        # nn.Parameter would bypass this scheme)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor,
                pad: torch.Tensor, proprio: torch.Tensor,
                **unused: Any):
        """image: (B, 3, H, W) uint8 or float.
        Returns (action (B,3) = dq1, dq2, grip; ball_off (B,2) | None; bin_off (B,2) | None).
        ball_off / bin_off are None when config.use_aux_heads is False."""
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        B = image.shape[0]
        fmap = self.vision(image)                              # (B, C, s, s)
        vis = self.vis_proj(fmap.flatten(2).transpose(1, 2))   # (B, n_vis, d_model)
        prop_tok = self.proprio_proj(proprio).unsqueeze(1)     # (B, 1, d_model)
        query = self.action_query.weight.unsqueeze(0).expand(B, 1, -1)
        # Fused LM: [vision | text | proprio | action-query] -> hidden states
        h = self.lm(tokens, pad, vis, torch.cat([prop_tok, query], dim=1))
        # Action head consumes the LM output (hidden state at the action-query token)
        a = self.action_head(h[:, -1, :])
        if not getattr(self.config, "use_aux_heads", True):
            return a * self.action_scale, None, None
        # Auxiliary grounding readouts over the fused vision-token states
        n_vis = vis.shape[1]
        states = h[:, :n_vis, :]
        ball_world = self.grounding(states)
        bin_world = self.bin_grounding(states)
        ee_world = proprio[:, 2:4] * EE_NORM
        ball_off = (ball_world - ee_world) / self.config.world_extent
        bin_off = (bin_world - ee_world) / self.config.world_extent
        return a * self.action_scale, ball_off, bin_off

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---------- HF-format save/load (weights + config + tokenizer in one dir) ----------
    def save_hf(self, path: str, tokenizer: "PocketVLATokenizer"):
        self.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)

    @classmethod
    def load_hf(cls, path: str, device: torch.device) -> tuple["PocketVLA", "PocketVLATokenizer"]:
        model = PocketVLA.from_pretrained(path).to(device).eval()
        tok = PocketVLATokenizer.from_pretrained(path)
        return model, tok


class PocketVLATokenizer(PreTrainedTokenizer):
    # "input_ids" must come first so transformers' padding/truncation resolves;
    # the model takes an explicit pad mask, no attention_mask needed.
    model_input_names = ["input_ids", "tokens"]

    def __init__(self, vocab: dict[str, int] | None = None, **kwargs):
        self.vocab = dict(vocab) if vocab else self._default_vocab()
        self.id2w = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        super().__init__(**kwargs)

    @staticmethod
    def _default_vocab() -> dict[str, int]:
        from ..lang.tokenizer import TOKENIZER
        return dict(TOKENIZER.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id2w.get(index, "<unk>")

    def build_inputs(self, text: str, lang_len: int) -> dict[str, Any]:
        """Text -> model inputs (padded token id array)."""
        ids = [self._convert_token_to_id(t) for t in self._tokenize(text)]
        arr = np.zeros(lang_len, dtype=np.int64)
        n = min(len(ids), lang_len)
        arr[:n] = ids[:n]
        return {"tokens": arr}

    def decode_tokens(self, tokens) -> str:
        return " ".join(self.id2w.get(int(t), "<unk>")
                        for t in tokens if int(t) != self.pad_id)

    # ---- Store the small vocab in the "vocab" field of tokenizer_config.json ----
    def save_vocabulary(self, save_directory: str,
                       filename_prefix: str | None = None) -> tuple[str]:
        import os
        path = os.path.join(save_directory, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=1)
        return (path,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import os
        vocab_path = os.path.join(str(pretrained_model_name_or_path), "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                vocab = json.load(f)
            kwargs["vocab"] = {k: int(v) for k, v in vocab.items()}
        else:
            kwargs.setdefault("vocab", None)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


# AutoClass registration: AutoModel.from_pretrained(..., trust_remote_code=True)
AutoConfig.register("pocketvla", PocketVLAConfig)
AutoModel.register(PocketVLAConfig, PocketVLA)
AutoTokenizer.register(PocketVLAConfig, slow_tokenizer_class=PocketVLATokenizer)
