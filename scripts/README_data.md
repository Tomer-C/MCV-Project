# Getting CelebA

`torchvision.datasets.CelebA(download=True)` pulls from Google Drive, which is
rate-limited and fails often ("Google Drive quota exceeded", or a missing
`gdown` dependency). Two reliable routes:

## Option A — let torchvision try (needs gdown)

```bash
pip install gdown
python -c "from torchvision.datasets import CelebA; CelebA('./data', download=True)"
```

## Option B — Kaggle (recommended, more reliable)

```bash
pip install kaggle
# put your kaggle.json in ~/.kaggle/ first
kaggle datasets download -d jessicali9530/celeba-dataset -p ./data --unzip
```

Then arrange the files so torchvision finds them:

```
data/
└── celeba/
    ├── img_align_celeba/          # 202599 jpgs
    ├── list_attr_celeba.txt
    ├── list_eval_partition.txt
    ├── identity_CelebA.txt
    ├── list_bbox_celeba.txt
    └── list_landmarks_align_celeba.txt
```

The Kaggle archive names the attribute files as `.csv`. Convert them:

```bash
cd data/celeba
python - <<'PY'
import csv, pathlib
# list_attr_celeba.csv -> the whitespace format torchvision expects
rows = list(csv.reader(open('list_attr_celeba.csv')))
hdr, body = rows[0][1:], rows[1:]
with open('list_attr_celeba.txt','w') as f:
    f.write(f"{len(body)}\n{' '.join(hdr)}\n")
    for r in body:
        f.write(r[0] + ' ' + ' '.join(f"{int(v):d}" if int(v)>0 else "-1" for v in r[1:]) + '\n')
print('wrote list_attr_celeba.txt')
PY
```

Verify with:

```bash
python scripts/inspect_attributes.py --data_root ./data
```

If that prints attribute counts, the dataset is wired up correctly.

## Skipping the download

`scripts/smoke_test.py` and `scripts/integration_test.py` need no data at all —
use them to validate code changes before you have CelebA in place.
