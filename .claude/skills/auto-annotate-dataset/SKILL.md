---
name: auto-annotate-dataset
description: >-
  Auto-annotate a folder of raw images into YOLO-format labels with a custom,
  user-defined class list, without drawing any bounding boxes by hand, using an
  open-vocabulary zero-shot detector (Grounding DINO). Use this whenever the user
  wants to label a new dataset but the categories aren't fixed yet (they depend on
  what's actually in the images) so the existing CarDD-trained model can't be reused
  for pseudo-labeling.
---

# Auto-annotate a new dataset with custom categories (no manual box-drawing)

## When to use this vs. `train-yolov8-cardd`

`train-yolov8-cardd` assumes CarDD's fixed 6-class taxonomy and already-labeled data.
Use *this* skill instead when the images are unlabeled and the class list isn't decided
yet — e.g. `Datasets/Hotai_iRent_cars/` (293 "Raw" + 240 "Annotated" real iRent photos;
despite the folder name, **none of them actually have label files** — verified by
searching for `.txt`/`.json`/`.xml` annotations under that tree and finding none). Once
this skill produces a trained model with its own classes, wiring it into the API follows
the same pattern as the CarDD skill (step 8 below).

## Facts that shape this (verified — re-check if the dataset or repo has moved)

- The API's `Detection` schema (`app/schemas/common.py`) has no mask field, so plain
  **Grounding DINO** (boxes only) is enough — no need for the heavier Grounded-SAM, which
  adds segmentation masks nothing downstream uses.
- `Datasets/Hotai_iRent_cars/Raw/` is split into `進行索賠資料` (claim filed) and
  `沒有進行索賠` (no claim filed) subfolders. This is a useful free sanity check: images
  under `沒有進行索賠` that end up with detected damage boxes are the likeliest false
  positives and should be reviewed first.
- The source images mix `.jpg`/`.jfif` extensions and can repeat filenames across
  subfolders — `scripts/auto_annotate.py` re-encodes every image to a uniquely-named
  `.jpg` via PIL before running detection, so this doesn't need to be handled manually.
- None of `autodistill`/`autodistill-grounding-dino`/`supervision` are in
  `requirements.txt` or `requirements-dev.txt` — they're one-off offline tooling, not an
  API runtime or test dependency, so they live in `requirements-autolabel.txt` instead.

## Steps

### 1. Decide the category list by looking at real images
Open a handful of images from both `Datasets/Hotai_iRent_cars/Raw/進行索賠資料` and
`.../沒有進行索賠` (and `Annotated/` if relevant) and note the damage types actually
visible — this is the one unavoidable human judgment call (naming classes), not
box-drawing. Don't assume CarDD's classes apply; write down whatever's actually there
(e.g. `scratch`, `dent`, `rust`, `crack`, `glass_shatter`, `lamp_broken`, ...).

### 2. Write the ontology
Copy `scripts/ontology.example.yaml` to `scripts/ontology.yaml` and edit it: each key is
a short natural-language prompt fed to Grounding DINO, each value is the class name that
lands in the generated YOLO labels. Multiple prompts may map to the same class.

### 3. Install dependencies
On Windows with a non-ASCII repo path (this one has Chinese characters), plain `pip
install` can fail building `rf-groundingdino` with `UnicodeDecodeError: 'cp950' codec
can't decode byte ...` — its setup script reads a non-ASCII file using the system
codepage. Force UTF-8 mode to fix it:
```bash
PYTHONUTF8=1 pip install -r requirements-autolabel.txt
```

### 4. Dry run on a small sample first
```bash
python scripts/auto_annotate.py \
  --input Datasets/Hotai_iRent_cars/Raw \
  --output Datasets/Hotai_yolo_draft_sample \
  --ontology scripts/ontology.yaml \
  --limit 20
```
Open a few of the resulting images alongside their `.txt` labels (or load the draft
folder into any YOLO-label viewer) and check the boxes look reasonable. Grounding DINO
prompts often need a round or two of rewording — e.g. "a dent on a car body panel" works
better than just "dent" — before behaving well. Re-run step 4 until satisfied.

### 5. Full run
```bash
python scripts/auto_annotate.py \
  --input Datasets/Hotai_iRent_cars/Raw \
  --output Datasets/Hotai_yolo_draft \
  --ontology scripts/ontology.yaml
```
Repeat for `Datasets/Hotai_iRent_cars/Annotated` if it should be included too (label into
a separate output folder, or merge afterward). The script prints per-class detection
counts and a list of zero-detection images to prioritize for review.

### 6. Review and correct a subset
First triage locally with `scripts/validate_annotations.py`, which draws the drafted
boxes onto the images and writes a stats report — no annotation tool needed yet:
```bash
python scripts/validate_annotations.py \
  --dataset Datasets/Hotai_yolo_draft \
  --claim-root Datasets/Hotai_iRent_cars/Raw
```
This writes annotated overlays to `<dataset>/review/` and a `review_report.txt` listing,
in priority order:
1. The zero-detection images.
2. Images under `沒有進行索賠` that got any detections (likely false positives).
3. Images with a suspiciously tiny or huge box (probable noise or a spurious full-image box).
4. Everything else, for a random spot-check.

Flip through `review/` in that order — either directly in a file explorer, or open
`web/review.html` in a browser (no server needed) and pick the dataset folder for a
filterable, tagged gallery with the same triage priority built in. Only the images that
actually need fixing then go into a YOLO-aware annotation tool — CVAT, Label Studio, or
makesense.ai all support importing existing YOLO labels for correction rather than
starting from a blank slate. Fix only what's wrong, then re-export back into the same
`images/`/`labels/` layout.

### 7. Split and train
The draft dataset's `data.yaml` points `train` and `val` at the same `images/` folder —
it's a review draft, not a training split. After corrections, split into proper
train/val folders, then train the same way as `train-yolov8-cardd` step 4:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="<corrected>/data.yaml", epochs=100, imgsz=512, batch=4, device=0)
```

### 8. Wire into the API (only once satisfied with the new classes)
Add the new category names to `DamageClass` in `app/schemas/common.py`.
`YOLOv8Evaluator`'s existing name-normalization (lowercase, spaces/hyphens → underscore)
already matches enum members automatically — no other code change needed, same as how
CarDD's 6 classes were wired in. Then follow `train-yolov8-cardd` steps 5–6 to install the
weights and verify `/api/v1/health` / `/api/v1/damage/evaluate` end-to-end.

## Reporting

State the final class list actually used (it may differ from the first draft after
prompt tuning), the per-class detection counts from the full run, and how many images
needed manual correction versus being accepted as-is.
