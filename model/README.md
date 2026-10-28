Put your exported files here (they are not committed):

- `model.bin`     PLE1 int4 model from `train/export_ple.py`
- `tokenizer.bin` decode table from `train/export_ple.py`
- `golden.txt`    prompt + expected greedy tokens from `export_ple.py --check`
- `golden_i8.txt` the same for int8 activations: `export_ple.py --regolden model --act i8`
- `kvec.bin`      kernel test vectors: `tools/kernel_vectors.py model/model.bin model/kvec.bin`

Every platform build reads them from this folder.
