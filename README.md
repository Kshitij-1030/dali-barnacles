# Counting barnacles: DALI Data Challenge

Detecting and counting barnacles in tide-pool photos from two hand-labeled photos (`img1`, `img2`).
The analysis is in [`barnacles.ipynb`](barnacles.ipynb). The helper functions (ground truth from the masks, scorer and chance baseline, patch classifier, peak finding, frame detection) are in [`barnacle_utils.py`](barnacle_utils.py), which the notebook imports.

## What I did

1. Turned the hand-drawn outlines into ground-truth barnacle centers (1500 in `img1`, 390 in `img2`).
2. Built a scorer (precision, recall, F1, count error) and a fair chance baseline: random points inside the labeled area.
3. Tried four detectors: brightness threshold, template matching, blob detection (difference of Gaussians), and a random-forest patch classifier.
4. Ran the best one on the two unlabeled photos, counting only inside the inner square of the wire frame, and checked the count by hand on a few crops.

## Results

Scored on the right half of each photo (trained on the left half). Chance is the F1 of random points in the same box.

| Method | `img1` right half: F1 (chance) | `img2` right half: F1 (chance) |
|---|---|---|
| Brightness threshold | unusable | not scored |
| Template matching | 0.43 (0.20) | not tested |
| Blob detection (DoG) | 0.26 (0.17) | not tested |
| Random forest, balanced negatives | about 0.73 (0.20) | about 0.56 (0.18) |

On a fully unseen labeled photo (`img2` with a classifier trained only on `img1`) the forest scores only about 0.38 at threshold 0.1 and overcounts by 65% there. At threshold 0.2 the count error drops to +8.5%, but the F1 is also lower (0.32), so the total depends heavily on the threshold.

On the unlabeled photos it detects about 990 barnacles inside the inner square of `unseen_img1` and about 83 on `unseen_img2`. A hand count on five random 200 x 200 crops matched the detector in total (87 vs 87), though individual crops were off by up to 36%. See the notebook for the caveats (incomplete labels, only two scenes, thresholds chosen on the test halves).

## How to run

1. Download the challenge photos and masks (`img1.png`, `img2.png`, `mask1.png`, `mask2.png`, `unseen_img1.png`, `unseen_img2.png`) and put them next to the notebook and `barnacle_utils.py`. They are not included in this repo.
2. `pip install -r requirements.txt`
3. Open `barnacles.ipynb` and choose *Restart and Run All*. The random-forest sections take a few minutes. All random choices are seeded, so numbers should reproduce to about the second decimal.