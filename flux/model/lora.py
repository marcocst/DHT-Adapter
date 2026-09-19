import copy
import gc
import math
import tqdm.autonotebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import FluxAttnProcessor2_0


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, do_training=True, sig_type=None, original_layer=None):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank))
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype)) * mask.to(hidden_states.device)
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)


class BSALinearLayer(nn.Module):
    """A frozen B/A basis with a trainable rank-by-rank middle matrix S.

    The effective adapter update is

        B M(t) (S - S_ref) M(t) A x,

    where M(t) is the timestep-dependent rank mask.  A and B are always
    frozen.  Only S is trainable.  ``S_ref`` makes both supported
    initializations exact no-ops at initialization for every timestep.
    """

    SUPPORTED_INITIALIZATIONS = {"random", "svd", "checkpoint"}

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        do_training=True,
        sig_type=None,
        original_layer=None,
        init_mode="random",
        svd_oversample=4,
        svd_niter=4,
        svd_device="auto",
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"BSA rank {rank} must be less or equal than "
                f"{min(in_features, out_features)}"
            )
        if init_mode not in self.SUPPORTED_INITIALIZATIONS:
            raise ValueError(
                f"Unsupported BSA initialization {init_mode!r}. Expected one of "
                f"{sorted(self.SUPPORTED_INITIALIZATIONS)}."
            )

        self.rank = rank
        self.init_mode = init_mode
        self.down = nn.Linear(in_features, rank, bias=False)   # A
        self.middle = nn.Linear(rank, rank, bias=False)        # S
        self.up = nn.Linear(rank, out_features, bias=False)    # B
        self.register_buffer(
            "middle_reference",
            torch.zeros(rank, rank),
            persistent=True,
        )

        if init_mode == "random":
            self._init_random()
        elif init_mode == "svd":
            if original_layer is None or not hasattr(original_layer, "weight"):
                raise ValueError(
                    "BSA SVD initialization requires the original nn.Linear layer "
                    "whose W0.weight will be decomposed."
                )
            self._init_from_weight_svd(
                original_layer.weight,
                oversample=svd_oversample,
                niter=svd_niter,
                svd_device=svd_device,
            )
        else:
            # Inference restores every tensor from lora.pt, so recomputing SVD or
            # relying on RNG here would be both expensive and less reproducible.
            self._init_checkpoint_placeholder()

        self.down.requires_grad_(False)
        self.up.requires_grad_(False)
        self.middle.requires_grad_(True)

    @torch.no_grad()
    def _init_random(self):
        # Both frozen bases must be non-zero; otherwise dL/dS would be zero.
        nn.init.normal_(self.down.weight, std=1 / self.rank)
        nn.init.normal_(self.up.weight, std=1 / self.rank)
        nn.init.zeros_(self.middle.weight)
        self.middle_reference.zero_()

    @torch.no_grad()
    def _init_checkpoint_placeholder(self):
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.middle.weight)
        nn.init.zeros_(self.up.weight)
        self.middle_reference.zero_()

    @torch.no_grad()
    def _init_from_weight_svd(self, weight, oversample, niter, svd_device):
        if oversample < 1:
            raise ValueError("svd_oversample must be at least 1.")
        if niter < 0:
            raise ValueError("svd_niter must be non-negative.")

        if svd_device == "auto":
            target_device = weight.device
        elif svd_device == "cpu":
            target_device = torch.device("cpu")
        elif svd_device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("bsa_svd_device='cuda' requires CUDA.")
            target_device = torch.device("cuda")
        else:
            raise ValueError("bsa_svd_device must be one of: auto, cpu, cuda.")

        work_weight = weight.detach().to(device=target_device, dtype=torch.float32)
        min_dim = min(work_weight.shape)
        q = min(max(self.rank, self.rank * oversample), min_dim)

        # This is a decomposition of the pretrained projection W0 itself.  No
        # gradient matrix is collected or decomposed.
        U, singular_values, V = torch.svd_lowrank(
            work_weight,
            q=q,
            niter=niter,
        )
        U_r = U[:, : self.rank]
        Vh_r = V[:, : self.rank].transpose(0, 1)
        S_r = torch.diag(singular_values[: self.rank])

        self.up.weight.copy_(U_r.to(self.up.weight))
        self.down.weight.copy_(Vh_r.to(self.down.weight))
        self.middle.weight.copy_(S_r.to(self.middle.weight))
        self.middle_reference.copy_(S_r.to(self.middle_reference))

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones(
                (1, self.rank),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        orig_dtype = hidden_states.dtype
        dtype = self.middle.weight.dtype
        mask = mask.to(device=hidden_states.device, dtype=dtype)

        # A and B are frozen, while S-S_ref is the trainable effective update.
        down_hidden_states = self.down(hidden_states.to(dtype))
        down_hidden_states = down_hidden_states * mask
        middle_delta = self.middle.weight - self.middle_reference
        middle_hidden_states = F.linear(down_hidden_states, middle_delta)
        middle_hidden_states = middle_hidden_states * mask
        up_hidden_states = self.up(middle_hidden_states)

        return up_hidden_states.to(orig_dtype)


class FluxLoraAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size,
        lora_linear_layer=LoRALinearLayer,
        cross_attention_dim=None,
        rank=4,
        do_training=True,
        sig_type='last',
        original_layer=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.do_training = do_training
        if original_layer:
            self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_q)
            self.to_k_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_k)
            self.to_v_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_v)
            if getattr(original_layer, "to_out", None):
                self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_out[0])

            if getattr(original_layer, "add_q_proj", None) is not None:
                self.to_q_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.add_q_proj)
                self.to_k_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.add_k_proj)
                self.to_v_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.add_v_proj)
        else:
            self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_k_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_v_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)

            self.to_q_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_k_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_v_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        scale=1.0,
        sigma_mask=None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states, sigma_mask)
        key = attn.to_k(hidden_states) + scale * self.to_k_lora(hidden_states, sigma_mask)
        value = attn.to_v(hidden_states) + scale * self.to_v_lora(hidden_states, sigma_mask)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states) + scale * self.to_q_proj_lora(encoder_hidden_states, sigma_mask)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states) + scale * self.to_k_proj_lora(encoder_hidden_states, sigma_mask)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states) + scale * self.to_v_proj_lora(encoder_hidden_states, sigma_mask)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states, sigma_mask)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim,
    ori_attn_proc,
    original_layer,
):
    return ori_attn_proc


def set_flux_transformer_attn_processor(
    transformer,
    set_attn_proc_func=default_set_attn_proc_func,
    set_attn_module_names=None,
):
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in tqdm.tqdm(transformer.attn_processors.items()):
        dim_head = transformer.config.attention_head_dim
        num_heads = transformer.config.num_attention_heads
        if name.endswith("attn.processor"):
            attention_module_name = name[: -len(".processor")]
            original_layer = transformer.get_submodule(attention_module_name)
            attn_procs[name] = (
                set_attn_proc_func(
                    name,
                    dim_head,
                    num_heads,
                    attn_processor,
                    original_layer,
                )
                if do_set_processor(name, set_attn_module_names)
                else attn_processor
            )

    transformer.set_attn_processor(attn_procs)
