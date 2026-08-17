import torch
import torch.nn as nn

from config import (
    VOCAB_SIZE,
    CONTEXT_SIZE,
    LAYERS,
    HEADS,
    EMBEDDING,
    DROPOUT
)


class Attention(nn.Module):

    def __init__(self):

        super().__init__()

        self.qkv = nn.Linear(
            EMBEDDING,
            EMBEDDING * 3
        )

        self.output = nn.Linear(
            EMBEDDING,
            EMBEDDING
        )

        self.heads = HEADS
        self.head_dim = EMBEDDING // HEADS

        mask = torch.tril(
            torch.ones(
                CONTEXT_SIZE,
                CONTEXT_SIZE
            )
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(
            x
        ).chunk(3, dim=-1)

        q = q.view(
            B, T,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / (
            self.head_dim ** 0.5
        )

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        result = weights @ v

        result = result.transpose(
            1, 2
        ).contiguous()

        result = result.view(
            B, T, C
        )

        return self.output(result)


class Block(nn.Module):

    def __init__(self):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            EMBEDDING
        )

        self.attention = Attention()

        self.norm2 = nn.LayerNorm(
            EMBEDDING
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                EMBEDDING,
                EMBEDDING * 4
            ),

            nn.GELU(),

            nn.Linear(
                EMBEDDING * 4,
                EMBEDDING
            )
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class NEXUS(nn.Module):

    def __init__(self):

        super().__init__()

        self.tokens = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING
        )

        self.positions = nn.Embedding(
            CONTEXT_SIZE,
            EMBEDDING
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(LAYERS)
            ]
        )

        self.norm = nn.LayerNorm(
            EMBEDDING
        )

        self.output = nn.Linear(
            EMBEDDING,
            VOCAB_SIZE,
            bias=False
        )

        self.output.weight = (
            self.tokens.weight
        )

    def forward(
        self,
        tokens,
        targets=None
    ):

        B, T = tokens.shape

        positions = torch.arange(
            T,
            device=tokens.device
        )

        x = (
            self.tokens(tokens)
            +
            self.positions(positions)
        )

        x = self.blocks(x)

        x = self.norm(x)

        logits = self.output(x)

        loss = None

        if targets is not None:

            loss = nn.functional.cross_entropy(
                logits.reshape(
                    -1,
                    VOCAB_SIZE
                ),
                targets.reshape(-1)
            )

        return logits, loss
