# experiments/tokenizer_v0.py

MASK_ID = 256
PAD_ID = 257
BOS_ID = 258
EOS_ID = 259

VOCAB_SIZE = 260


class ByteTokenizer:
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Transforme une chaîne Python en IDs de tokens.

        Les IDs 0-255 correspondent directement aux bytes UTF-8.
        """

        tokens = list(text.encode("utf-8"))

        if add_bos:
            tokens.insert(0, BOS_ID)

        if add_eos:
            tokens.append(EOS_ID)

        return tokens

    def decode(
        self,
        tokens: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Transforme des IDs de bytes en texte UTF-8.
        """

        if skip_special_tokens:
            tokens = [
                token_id
                for token_id in tokens
                if 0 <= token_id <= 255
            ]

        byte_sequence = bytes(tokens)

        return byte_sequence.decode(
            "utf-8",
            errors="replace",
        )


if __name__ == "__main__":
    tokenizer = ByteTokenizer()

    text = "café"

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    decoded = tokenizer.decode(tokens)

    print("Texte original :")
    print(text)

    print("\nTokens :")
    print(tokens)

    print("\nTexte reconstruit :")
    print(decoded)

    print("\nNombre de tokens :")
    print(len(tokens))

    assert decoded == text

    print("\nTest encode/decode réussi.")