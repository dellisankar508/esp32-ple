Put your exported files here (they are not committed):

- `model.bin`     PLE1 int4 model from `train/export_ple.py`
- `tokenizer.bin` decode table from `train/export_ple.py`
- `golden.txt`    prompt + expected greedy tokens from `export_ple.py --check`

Every platform build reads them from this folder.
