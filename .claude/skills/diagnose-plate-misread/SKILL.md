---
name: diagnose-plate-misread
description: >-
  Find out why the iRent license-plate recogniser reads a plate wrong (e.g. RDX-2376 stored as
  RQI-270), never reads it, or links a photo to the wrong car - by tracing the real EasyOCR reads
  through format_plate / resolve_plates, then proposing a concrete fix. Use this whenever the
  user says a plate is misidentified, misread, OCR'd wrong, "always comes out as" something
  else, not found, or when ai_anomaly_alerts / damage_annotations rows carry a plate that matches
  no vehicle - even if they just ask "why is X read as Y".
---

# Diagnose a plate misread

The goal is an evidence-backed answer to "why does photo P give plate Y instead of X", plus a
fix proposal. Guessing from the code alone is unreliable: the same text-rewriting rules can
explain several wrong plates, and EasyOCR's raw output changes with crop size and photo. So the
core of this skill is reproducing the misread on real photos and reading the raw OCR text.

Diagnose and propose; do not edit `app/` code until the user agrees to the fix.

## 1. Pin down the case

Establish the real plate (X), the wrong output (Y) and where Y showed up. Useful places to look
(read-only; never write to `data/irent_op_backend.sqlite` while diagnosing):

- `ai_anomaly_alerts.plate_number` / `damage_annotations.plate_number`: rows with Y, their
  `detected_at` / `case_id`, and `api_call_logs` around that time say which endpoint produced it.
- `vehicles.license_plate = X` gives the vehicle id; `vehicle_photos` has its stored photos
  (files under `data/vehicle_photos/<vehicle_id>/`).
- Dataset photos are named after their plate: `data/Hotai_iRent_cars/**/<X>*.jpg`, with or
  without the hyphen (`RCG2235右前 借.jfif`).

Know which path does OCR. `/plate/recognize` always does. `/damage/evaluate`, `/tire/detect` and
`/fee/estimate` take the plate from the `plate_number` form field, or else from the uploaded
file's name (`app/services/vehicle_link.py`). They never OCR. Rows written by those endpoints
before that change may still carry OCR misreads. A wrong plate from those endpoints today means a
wrong file name or form value, not OCR. Uploaded photos are **not stored**, so the exact image
behind an old row may be gone. In that case, trace the car's other photos and the dataset photo
of the same corner, and ask the user for the original if nothing reproduces Y.

## 2. Trace the real reads

Run the bundled tracer from the repository root:

```bash
# every photo of the car the repo knows about (stored corners + dataset files)
python .claude/skills/diagnose-plate-misread/scripts/trace_plate.py --plate RDX-2376
# specific photos
python .claude/skills/diagnose-plate-misread/scripts/trace_plate.py --expect RDX-2376 path/a.jpg
# no OCR: just how format_plate rewrites candidate raw strings (fast; good for hypotheses)
python .claude/skills/diagnose-plate-misread/scripts/trace_plate.py --text ROI270 R01270
```

For each photo it prints every raw read with its confidence and its source. The source is
`located crop` (plate locator or detector) or `whole-frame scan` (the fallback). It then prints
each `format_plate` step (cleaning, leading-noise strip, layout matched, O→Q, confusion swaps),
the votes, and the BEST plate with OK / MISMATCH. OCR runs on CPU at a few seconds per photo, so
it reads at most `--max 12` photos. If it says the mock recogniser is on, the result is
meaningless. Unset `IRENT_PLATE_USE_MOCK` instead of reasoning from fake plates.

Find the photo(s) whose BEST is Y, then the exact raw read that became Y. That one line plus its
steps is the core evidence. If no photo reproduces Y, say so plainly and use `--text` to show
which raw strings *would* produce Y. Label that as a hypothesis, not a finding.

## 3. Name the failing stage

Map the evidence to the stage that let the wrong plate through. Usually more than one stage is
involved: name the one that created Y and the one that should have stopped it. All of these are
in `app/services/plate_recognizer.py`:

| Symptom in the trace | Stage | Where |
|---|---|---|
| `confusion swap` turned digits into letters (0→Q, 1→I, 5→S, 8→B) or letters into digits | fallback re-segmentation in `format_plate` | `_TO_LETTER`, `_TO_DIGIT` |
| `letter 'O' forced to 'Q'` (often a misread D or 0) | `_FIX_LETTERS` | `format_plate` |
| `NOT the current ABC-1234 layout`: a glyph was dropped (2376 → 270) | `_PLATE_RE` accepts 2-3 letters + 3-4 digits | `_PLATE_RE` |
| `contains I/O`: Taiwan plates never use these letters, yet it was accepted | no letter check | `_PLATE_RE` / `format_plate` |
| a tiny-confidence read still won | valid-format reads bypass `plate_min_confidence`; votes and layout outrank confidence | `resolve_plates` |
| right text read once, wrong text read by more variants | vote counting across preprocessing variants | `resolve_plates`, `preprocess_plate` |
| K↔W (or H/N/V) swaps | glyph confusion at a crop width | `_LETTER_SWAPS`, `_CROP_WIDTHS` |
| only `whole-frame scan` reads, or none | the plate was never located (angle, glare, small) | `plate_locator.py`, `plate_detector_weights_path` |
| leading I/L/J / trailing junk glyph | plate frame or screw read as text | `_strip_leading_noise`, `_BORDER_TRIM_FRAC` |

Check the matching function in the current code before naming it. The table describes the code
as it was when this skill was written.

## 4. Propose a fix

Prefer fixes that reject impossible plates over fixes that hard-code the one case. A good
proposal names the change, shows why it rejects Y, and shows it still accepts X and the other
known plates. Ideas that tend to hold up:

- Reject results containing I or O, since no Taiwan plate uses them. Then the next candidate
  wins instead.
- Do not let the confusion-swap fallback or a below-`plate_min_confidence` read beat nothing.
  Returning "no plate" is better than returning a wrong plate that silently links a damage
  record to no car or the wrong car.
- Rank the current `ABC-1234` layout above shorter layouts when the photo is of a car.
- Check against the `vehicles` registry when a database is configured, and treat an
  unregistered plate as a low-confidence read.
- For operational flows, avoid OCR entirely: the client sends the plate or names the file after
  it (already the case for damage, tire and fee).

Before recommending a change, check its blast radius. Run the change through `--text` on the
offending raw strings. Then say how to confirm nothing else regresses:
`python scripts/evaluate_plate_recognition.py --csv runs/plate_eval_after.csv` compared with the
latest `runs/plate_eval*.csv`. Include a regression test in `tests/test_plate.py` that feeds the
offending raw read (e.g. `R01270` must not become `RQI-270`). If the user asks you to apply the
fix, also run the `run-damage-tests` skill.

## 5. Report

Keep it short and lead with the answer:

1. **Cause** in one or two sentences, e.g. *"On the cracked rear-right photo EasyOCR read
   `R01270` (10 %). format_plate's confusion swap turned `01` into `QI`, giving `RQI-270`, and
   it was the only valid-looking read, so it won despite its 10 % confidence."*
2. **Evidence**: the photo path and the trace lines that show it. If the misread does not
   reproduce, say that.
3. **Why it got through**: the stage(s) from step 3, with `file:line`.
4. **Proposed fix** and how to verify it (regression test + benchmark).
5. **Side effects worth knowing**: e.g. existing DB rows still carrying Y. Offer to clean them up
   rather than touching them.
