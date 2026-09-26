# Packer determinism across machines

The format 2 pack of google/gemma-4-12B-it (revision 707f0a3), made on the Mac (Apple M4 Max, arm64, `binade/pack.py` @ `bd1ba8b`) and on the GCP VM (AMD EPYC 9B45, x86_64, `binade/pack.py` @ `235ff64`, same packer code), is byte-identical:

    shasum -a 256 models/gemma-4-12B-it-binade-v2/model.safetensors   # Mac
    sha256sum     models/gemma-4-12B-it-binade-v2/model.safetensors   # VM
    2dc23fbeec9c2feded3215603cb1711ef6e64723646d276ec66713b8fa4441a6   (both)

The 26B packs agree in size and bits per weight (35.05 GB, 10.747); their hashes were not compared.
