# Checkpoint inventory

Checkpoint binaries are stored in the project's verified backup. Public weight
downloads are currently unavailable; the table below lists checkpoint identifiers.

| Archived model | SHA256 of full checkpoint |
|---|---|
| Teacher800 cosine recovery | `1626c57905203cd9d4f94bd2f9f0f90b580d720db1e6d2bbf630b4c792935729` |
| Pure student20 from that teacher | `ad15ef89f9d562736b6d9f2210525c12e7f70e0f0f1d1205906638631e03c6f8` |
| Direct100 | `53ad3147d9ceca605ad1627517abab30b217dda38612bd92c0820ffe7a01d0d9` |

Full training checkpoints use PyTorch serialization and can contain executable
pickle payloads. Only load trusted files after checking their hashes. The export
utility requires explicit `--trust-checkpoint`, reads the source without changing
it, and emits a self-contained EMA inference package with its own SHA256. Inference uses
`weights_only=True`. Full optimizer/RNG states are needed for exact continuation,
but not for inference. Check third-party terms before distributing any weights.
