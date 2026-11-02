# TODO

- **Pack the vision and audio towers (the `mm` class).** They are not on the text decode path and have no exact kernels, so a pack would rebuild them in BF16 at load (exact by construction); the gain is file size only, about 0.6 GB on Gemma 4 E4B.
