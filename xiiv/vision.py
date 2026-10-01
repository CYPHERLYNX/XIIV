"""Minimal image preprocessing for the XIIV multimodal scaffold.

The scaffold is ARCHITECTURE-READY, not trained: PatchEmbed exists in
xiiv/model.py and the sampler accepts an image prefix, but no image-text
training has been run (that needs captioned data + a GPU — see the
10pm roadmap). Everything here is dependency-light (torch + PIL only)
and shape-tested.
"""
import torch


def load_image_pixels(path: str, image_size: int = 64) -> torch.Tensor:
    """Load an image file -> (3, S, S) float32 tensor in [0, 1].

    Raises a clear error if Pillow is missing instead of failing obscurely.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "xiiv.vision needs Pillow: pip install pillow") from e
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    px = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0
    return px.permute(2, 0, 1).contiguous()  # (3, S, S)


@torch.no_grad()
def encode_image(model, pixels: torch.Tensor) -> torch.Tensor:
    """pixels: (3, S, S) or (B, 3, S, S) -> (N, d) or (B, N, d) prefix embeds.

    Requires a model built with cfg.image_enabled=True.
    """
    if model.patch_embed is None:
        raise ValueError("model was built with cfg.image_enabled=False")
    one = pixels.dim() == 3
    if one:
        pixels = pixels.unsqueeze(0)
    out = model.patch_embed(pixels.to(next(model.parameters()).device))
    return out.squeeze(0) if one else out


def make_multimodal_canvas(n_image_tokens: int, text_ids: torch.Tensor,
                           mask_id: int):
    """Build a sampler canvas for image-conditioned generation.

    Layout: [N frozen image-prefix positions (mask_id placeholders)]
            + [text_ids, where mask_id marks positions to fill].

    Returns (prompt_ids, frozen): pass both to sample_trajectory along
    with image_embeds=encode_image(...) — the model substitutes the
    prefix embeddings in place, so text positions keep their indices.
    """
    n = int(n_image_tokens)
    prefix = torch.full((n,), mask_id, dtype=torch.long)
    prompt_ids = torch.cat([prefix, text_ids.to(torch.long)])
    frozen = torch.zeros(len(prompt_ids), dtype=torch.bool)
    frozen[:n] = True  # the image prefix is never denoised
    return prompt_ids, frozen
