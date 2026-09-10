import torch


def apply_rope(
    x: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Applique Rotary Positional Embeddings.

    x :
        [B, H, T, D]

    Retour :
        [B, H, T, D]
    """

    if x.ndim != 4:
        raise ValueError(
            "RoPE attend un tensor [B, H, T, D]."
        )

    B, H, T, D = x.shape

    if D % 2 != 0:
        raise ValueError(
            "head_dim doit être pair pour RoPE."
        )

    # -----------------------------------
    # 1. Fréquences de rotation
    # -----------------------------------

    dimension_indices = torch.arange(
        0,
        D,
        2,
        device=x.device,
        dtype=torch.float32,
    )

    inverse_frequencies = (
        1.0
        / (
            base
            ** (
                dimension_indices
                / D
            )
        )
    )

    # Shape :
    # [D / 2]

    # -----------------------------------
    # 2. Positions
    # -----------------------------------

    positions = torch.arange(
        T,
        device=x.device,
        dtype=torch.float32,
    )

    # Shape :
    # [T]

    # -----------------------------------
    # 3. Angles
    # -----------------------------------

    angles = torch.outer(
        positions,
        inverse_frequencies,
    )

    # Shape :
    # [T, D/2]

    cos = angles.cos()
    sin = angles.sin()

    # Pour permettre le broadcasting :
    #
    # [T,D/2]
    #     ↓
    # [1,1,T,D/2]

    cos = cos[
        None,
        None,
        :,
        :
    ].to(x.dtype)

    sin = sin[
        None,
        None,
        :,
        :
    ].to(x.dtype)

    # -----------------------------------
    # 4. Séparer dimensions paires/impaires
    # -----------------------------------

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Chacun :
    # [B,H,T,D/2]

    # -----------------------------------
    # 5. Rotation
    # -----------------------------------

    rotated_even = (
        x_even * cos
        - x_odd * sin
    )

    rotated_odd = (
        x_even * sin
        + x_odd * cos
    )

    # -----------------------------------
    # 6. Recombiner
    # -----------------------------------

    rotated = torch.stack(
        (
            rotated_even,
            rotated_odd,
        ),
        dim=-1,
    )

    # [B,H,T,D/2,2]

    rotated = rotated.flatten(
        -2
    )

    # [B,H,T,D]

    return rotated


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    x = torch.randn(
        2,
        4,
        8,
        16,
        device=device,
    )

    y = apply_rope(x)

    print("Input shape :")
    print(x.shape)

    print("\nOutput shape :")
    print(y.shape)

    print("\nPremier vecteur avant RoPE :")
    print(x[0, 0, 0])

    print("\nMême vecteur position 0 après RoPE :")
    print(y[0, 0, 0])

    print("\nVecteur position 1 après RoPE :")
    print(y[0, 0, 1])


if __name__ == "__main__":
    main()