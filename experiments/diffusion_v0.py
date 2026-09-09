import torch

from tokenizer_v0 import (
    ByteTokenizer,
    MASK_ID,
    PAD_ID,
    BOS_ID,
    EOS_ID,
)

def sample_timesteps(
    batch_size: int,
    device: torch.device | str,
    min_t: float = 0.05,
    max_t: float = 0.95,
) -> torch.Tensor:

    random_t = torch.rand(
        batch_size,
        device=device,
    )

    timesteps = (
        min_t
        +
        random_t
        * (max_t - min_t)
    )

    return timesteps

def corrupt_batch(
    clean_tokens: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Ajoute du bruit à un batch de tokens en remplaçant
    certains tokens par MASK_ID.

    clean_tokens:
        shape [B, T]

    timesteps:
        shape [B]
        une valeur t entre 0 et 1 pour chaque séquence.

    Retourne:
        noisy_tokens:
            shape [B, T]

        corruption_mask:
            shape [B, T]
            True aux positions qui ont été masquées.
    """

    if clean_tokens.ndim != 2:
        raise ValueError(
            "clean_tokens doit avoir la shape [B, T]"
        )

    if timesteps.ndim != 1:
        raise ValueError(
            "timesteps doit avoir la shape [B]"
        )

    batch_size, sequence_length = clean_tokens.shape

    if timesteps.shape[0] != batch_size:
        raise ValueError(
            "Il faut un timestep par séquence du batch."
        )

    # [B] -> [B, 1]
    #
    # Cela permettra à PyTorch de comparer chaque ligne
    # avec le timestep correspondant.
    mask_probabilities = timesteps.unsqueeze(1)

    # Génère un nombre aléatoire entre 0 et 1
    # pour chaque token.
    #
    # Shape :
    # [B, T]
    random_values = torch.rand(
        clean_tokens.shape,
        device=clean_tokens.device,
    )

    # Exemple :
    #
    # random_values = 0.23
    # t = 0.50
    #
    # 0.23 < 0.50
    # donc ce token sera masqué.
    corruption_mask = (
        random_values
        < mask_probabilities
    )

    # Certains tokens spéciaux ne doivent pas être détruits.
    protected_tokens = (
        (clean_tokens == PAD_ID)
        | (clean_tokens == BOS_ID)
        | (clean_tokens == EOS_ID)
    )

    corruption_mask = ensure_at_least_one_mask(
    corruption_mask,
    clean_tokens,
    )   

    # Copie obligatoire :
    # on ne veut pas modifier clean_tokens.
    noisy_tokens = clean_tokens.clone()

    noisy_tokens[corruption_mask] = MASK_ID

    return noisy_tokens, corruption_mask


def visualize_tokens(
    tokenizer: ByteTokenizer,
    tokens: list[int],
) -> str:
    """
    Affiche les tokens spéciaux sous forme lisible.

    Exemple :
    [MASK]
    [BOS]
    [EOS]
    """

    pieces = []
    byte_buffer = []

    def flush_bytes():
        if byte_buffer:
            text = bytes(
                byte_buffer
            ).decode(
                "utf-8",
                errors="replace",
            )

            pieces.append(text)
            byte_buffer.clear()

    for token_id in tokens:

        if 0 <= token_id <= 255:
            byte_buffer.append(token_id)
            continue

        flush_bytes()

        if token_id == MASK_ID:
            pieces.append("[MASK]")

        elif token_id == PAD_ID:
            pieces.append("[PAD]")

        elif token_id == BOS_ID:
            pieces.append("[BOS]")

        elif token_id == EOS_ID:
            pieces.append("[EOS]")

        else:
            pieces.append(
                f"[UNKNOWN:{token_id}]"
            )

    flush_bytes()

    return "".join(pieces)


def test_single_sentence():

    torch.manual_seed(42)

    tokenizer = ByteTokenizer()

    text = "The cat sleeps on the sofa."

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    clean_tokens = torch.tensor(
        [tokens],
        dtype=torch.long,
    )

    # Un seul élément dans le batch.
    #
    # Donc un seul timestep.
    timesteps = torch.tensor(
        [0.50],
        dtype=torch.float32,
    )

    noisy_tokens, corruption_mask = (
        corrupt_batch(
            clean_tokens,
            timesteps,
        )
    )

    print("Texte original :")

    print(
        visualize_tokens(
            tokenizer,
            clean_tokens[0].tolist(),
        )
    )

    print("\nTexte bruité :")

    print(
        visualize_tokens(
            tokenizer,
            noisy_tokens[0].tolist(),
        )
    )

    print("\nTimestep :")
    print(timesteps[0].item())

    print("\nNombre de tokens :")
    print(clean_tokens.shape[1])

    print("\nNombre de tokens masqués :")

    print(
        corruption_mask[0]
        .sum()
        .item()
    )

    print("\nShape clean_tokens :")
    print(clean_tokens.shape)

    print("\nShape corruption_mask :")
    print(corruption_mask.shape)

def test_noise_levels():

    tokenizer = ByteTokenizer()

    text = (
        "DiffuThink learns to reconstruct text."
    )

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    clean = torch.tensor(
        [tokens],
        dtype=torch.long,
    )

    noise_levels = [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90,
    ]

    for t in noise_levels:

        # Même seed à chaque fois :
        # pratique pour visualiser
        # l'augmentation du bruit.
        torch.manual_seed(42)

        timestep = torch.tensor(
            [t],
            dtype=torch.float32,
        )

        noisy, mask = corrupt_batch(
            clean,
            timestep,
        )

        masked_count = (
            mask.sum().item()
        )

        eligible_count = (
            (
                (clean != BOS_ID)
                & (clean != EOS_ID)
                & (clean != PAD_ID)
            )
            .sum()
            .item()
        )

        ratio = (
            masked_count
            / eligible_count
        )

        print("\n" + "=" * 70)

        print(
            f"t = {t:.2f}"
        )

        print(
            f"Masqué réellement : "
            f"{ratio:.2%}"
        )

        print(
            visualize_tokens(
                tokenizer,
                noisy[0].tolist(),
            )
        )

def test_timestep_sampling():

    batch_size = 8

    timesteps = sample_timesteps(
        batch_size=batch_size,
        device="cpu",
    )

    print("Timesteps :")

    print(timesteps)

    print("\nShape :")

    print(timesteps.shape)

    print("\nMinimum :")

    print(
        timesteps.min().item()
    )

    print("\nMaximum :")

    print(
        timesteps.max().item()
    )

def ensure_at_least_one_mask(
    corruption_mask: torch.Tensor,
    clean_tokens: torch.Tensor,
) -> torch.Tensor:

    batch_size = (
        clean_tokens.shape[0]
    )

    for batch_index in range(
        batch_size
    ):

        has_mask = (
            corruption_mask[
                batch_index
            ]
            .any()
        )

        if has_mask:
            continue

        eligible_positions = (
            (clean_tokens[batch_index] != PAD_ID)
            & (clean_tokens[batch_index] != BOS_ID)
            & (clean_tokens[batch_index] != EOS_ID)
        )

        eligible_indices = (
            torch.where(
                eligible_positions
            )[0]
        )

        if len(eligible_indices) == 0:
            continue

        random_index = torch.randint(
            low=0,
            high=len(
                eligible_indices
            ),
            size=(1,),
            device=clean_tokens.device,
        )

        chosen_position = (
            eligible_indices[
                random_index
            ]
        )

        corruption_mask[
            batch_index,
            chosen_position,
        ] = True

    return corruption_mask

if __name__ == "__main__":
    test_timestep_sampling()