## Homework 3 – Motion Estimation & Compensation

| Student ID | Name |
|-----------|-----------|
| 313553014 | Yi-Cheng Liao |

### Environments

```
# python 3.9
pip install -r requirements.txt
```

## Usage
Assuming two grayscale PNGs are available, e.g., `one_gray.png` (reference) and `two_gray.png` (target).

### Full Search (multiple ranges)
```bash
python vc_hw3_me_mc.py --ref one_gray.png --target two_gray.png --algo full --search 8 16 32
```

### Three-Step Search (R=8)
```bash
python vc_hw3_me_mc.py --ref one_gray.png --target two_gray.png --algo tss --search 8 16 32
```