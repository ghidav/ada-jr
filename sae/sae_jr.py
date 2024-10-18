import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn
import torch.nn.functional as F

from .config import JRSaeConfig
from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    pre_acts: Tensor
    """Activations of the top-k latents."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    l0_loss: list[Tensor]
    """L0 loss"""

    l1_loss: Tensor
    """L1 loss"""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""


class JRSae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: JRSaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

        self.threshold = nn.Parameter(
            torch.zeros(self.num_latents, dtype=dtype, device=device)
        )

        self.feature_mask = torch.nn.Parameter(torch.randn(self.num_latents, dtype=dtype, device=device))

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "JRSae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: JRSae.load_from_disk(repo_path / layer, device=device, decoder=decoder)
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: JRSae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "JRSae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return JRSae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "JRSae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = JRSaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = JRSae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic (adjusted for dtype)
        sae_in = x.to(self.dtype) - self.b_dec
        
        # Encoder forward pass
        out = self.encoder(sae_in)
        
        # Apply threshold to activations (ReLU followed by thresholding)
        pre_acts = F.relu(out) * (out > self.threshold)
        
        # Generate a stochastic mask for L0 regularization
        mask = self.concrete_sample(self.feature_mask, self.training)
        
        return pre_acts * mask

    def decode(self, pre_acts: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        return pre_acts @ self.W_dec + self.b_dec

    def concrete_sample(self, logits, training=True):
        gamma = -0.1
        zeta = 1.1
        beta = self.cfg.beta  # Should be set appropriately
        if training:
            noise = torch.rand_like(logits)
            z = torch.log(noise) - torch.log(1 - noise)
            y = torch.sigmoid((logits + z) / self.cfg.temperature)
            stretched_y = y * (zeta - gamma) + gamma
            return torch.clamp(stretched_y, 0, 1)
        else:
            hard_concrete = (logits > 0).float()
            return hard_concrete
    
    def l0_loss(self):
        gamma = -0.1
        zeta = 1.1
        beta = self.cfg.beta  # Same as above
        temp = self.cfg.temperature
        logits = self.feature_mask
        # Compute the probability of non-zero gate
        pre_prob = (logits - temp * torch.log(torch.tensor(-gamma / zeta))).sigmoid()
        expected_l0 = pre_prob.mean()
        return expected_l0

    def forward(self, x: Tensor, dead_mask: Tensor | None = None) -> ForwardOutput:
        pre_acts = self.pre_acts(x)

        # Decode and compute residual
        sae_out = self.decode(pre_acts)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        l1_loss = pre_acts.abs().sum()
        real_l0_loss = (pre_acts != 0).type(torch.float32).sum(-1).mean()
        
        approx_l0_loss = self.l0_loss().sum(-1).mean()
    
        return ForwardOutput(
            sae_out,
            pre_acts,
            fvu,
            (real_l0_loss, approx_l0_loss),
            l1_loss,
            auxk_loss
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
