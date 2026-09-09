---
name: train-yolov8-cardd
description: >-
  Train a real YOLOv8 car-damage detector on the CarDD dataset and wire it into
  this API in place of the mock detector. Use this whenever the user asks to
  train a model, produce real weights, or replace/switch from the mock
  evaluator to a real YOLOv8 model.
---

# Train YOLOv8 on CarDD and wire it into the API

`build_evaluator()` in `app/services/evaluator.py` already contains full logic to load a
real `YOLOv8Evaluator` from `IRENT_YOLO_WEIGHTS_PATH` and falls back to the mock detector
only when that path is unset, missing, or fails to load. **No application code needs to
change for this task** — the missing piece is producing a trained `.pt` file from CarDD and
pointing the app at it.

## Facts that shape this (verified — re-check if the dataset or repo has moved)

- CarDD's COCO annotations (`CarDD_release/CarDD_COCO/annotations/instances_*.json`, inside
  the dataset zip, e.g. `CarDD_release.zip`) define exactly 6 categories: `dent, scratch,
  crack, glass shatter, lamp broken, tire flat` (2,816 train / 810 val / 374 test images,
  6,211 train annotations). Normalized (`" " → "_"`) these are `dent, scratch, crack,
  glass_shatter, lamp_broken, tire_flat` — all 6 already exist in `DamageClass`
  (`app/schemas/common.py`), so **no enum or label-mapping change is needed**;
  `YOLOv8Evaluator.__init__`'s existing normalization matches them automatically. The
  enum's extra `missing_part`/`paint_chip` values simply won't be produced by this model.
- `requirements.txt` already lists `ultralytics` and `numpy` (the "full install" extras) —
  if the active venv only has the light install, run
  `pip install -r requirements-dev.txt` before training.
- If training on a small-VRAM GPU (e.g. 2 GB), use the nano model (`yolov8n.pt`), a modest
  `imgsz` (e.g. 512), and a small `batch` (e.g. 4); fall back to `device="cpu"` if it still
  OOMs. Check available VRAM first with `nvidia-smi`.
- `tests/conftest.py` always overrides the evaluator dependency with `MockEvaluator`
  regardless of `IRENT_YOLO_WEIGHTS_PATH`, so the pytest suite is unaffected by any of this
  and needs no changes — it will keep passing before and after real weights are wired in.

## Steps

### 1. Install training/inference dependencies
```bash
pip install -r requirements-dev.txt   # pulls in ultralytics + torch (CUDA-enabled wheel on Windows)
python -c "import torch; print(torch.cuda.is_available())"   # confirm GPU visibility
```

### 2. Extract CarDD and convert COCO annotations to YOLO format
Extract the dataset zip to a working directory outside the git repo (it's large and not
source). Then convert the COCO json to YOLO `.txt` labels with Ultralytics' built-in
converter:
```python
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir="<extracted>/CarDD_release/CarDD_COCO/annotations",
    save_dir="<workdir>/CarDD_yolo",
    use_segments=False,   # bounding boxes only — the API's Detection schema has no mask field
)
```
This writes one labels subfolder per source JSON file, with 0-indexed class ids in
ascending category-id order (`0=dent … 5=tire_flat`). Arrange the final layout Ultralytics
expects — copy/symlink images next to the converted labels, renamed to the conventional
split names:
```
CarDD_yolo/
  images/{train,val,test}/*.jpg   <- from CarDD_COCO/{train2017,val2017,test2017}
  labels/{train,val,test}/*.txt   <- from convert_coco's per-split output, renamed
```

### 3. Write `data.yaml`
```yaml
path: <workdir>/CarDD_yolo
train: images/train
val: images/val
test: images/test
names:
  0: dent
  1: scratch
  2: crack
  3: glass_shatter
  4: lamp_broken
  5: tire_flat
```

### 4. Train
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=100, imgsz=512, batch=4, device=0)
```
Best weights land at `runs/detect/train/weights/best.pt`. If it OOMs even at
`batch=4`/`imgsz=512`, lower further (`batch=2`) or switch to `device="cpu"` (slower but
memory-unconstrained). Training elsewhere (e.g. Colab) and copying `best.pt` back works
fine too — step 5 onward doesn't care where training happened.

### 5. Install the weights and point the app at them
```bash
mkdir weights
cp runs/detect/train/weights/best.pt weights/car_damage.pt
```
In `.env` (copy from `.env.example` if it doesn't exist yet):
```
IRENT_YOLO_WEIGHTS_PATH=./weights/car_damage.pt
IRENT_YOLO_DEVICE=cpu        # or "0" for GPU inference
```

### 6. Verify end-to-end
- Start the app: `./.venv/Scripts/python.exe -m uvicorn app.main:app --reload`.
- `GET /api/v1/health` — confirm the reported detector is `yolov8`, not `mock-detector`;
  check the startup log for "Loaded YOLOv8 detector..." rather than a fallback warning.
- `POST /api/v1/damage/evaluate` with a real damaged-car photo (e.g. a held-out CarDD test
  image, or via `web/index.html`) and sanity-check the returned detections/boxes.
- Run `pytest` — should still pass unchanged (fixtures force the mock evaluator).

## Reporting

State plainly which class the app is actually running (`is_mock` in the response, or the
`/health` detector field) — do not report the switch as done unless `/health` shows the
real `yolov8` detector loaded.
