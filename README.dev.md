# TODOs

- 10/12/2022

  - [*] Auto-arranged windows across screen
  - [*] Debugged display bboxes on grayscale images (gray->color in vis) -> create new np.array that's 3-dim for each displayed video (was overwritting in place, but gray weren't right dims)
  - [*] Add out_format variable to filter classes
  - [*] Convert errode/dilate to cv2.opening (see docs)
    - [*] Erodes fish too much - also why are scale numbers not fully filtered by mean filter
    - [*] Not temporal and insufficient; other options in findpeaks package
  - [*] Made demos extensible to general filters
    - [*] demo: dft, mean, contour; run: dft ... refactored
  - [*] Test findpeaks filters
    - [*] VERY VERY SLOW (20s / frame?)
  - [*] Revisiting NLMeans filter from OpenCV -> keep; strong filter
    - [*] Slow, but rather effective
    - [*] Testing with turbine + intensity filter
      - [*] Intensity filter debug after refactors (use simple fixed thresh to start?) -> Basic hard threshold working pretty well
      - [*] NLMeans is the major bottleneck, but is helping a lot with noise despeckling

- 10/13/2022

  - [*] Implementing tracklet association filter -> only keep bounding boxes that are consistent
    - [] Only implemented for current-prior tracklet scale and distance confirmation->I'd love for it to be n-windowed and to associate across n-frames

- [] Ground filters in existent methods to reference and describe mathematically in paper
  - [] Background filter
  - [] Intensity filter?
- [*] Improve filter output (lots of noise/speckle)
  - [*] Implement speckle filter -> NlMeans but very slow
- [] Implement and test additional filters
  - [] Finish implementing PIV filter
- [] Batch videos that are too large
- [] Record params values in results
- [] Record timing of runs

- 10/18/2022

  - [*] Finalize experiments list
    - [ ] Frames removed metric (per frame)
    - [ ] False negative rate (per track)
    - [ ] T/F P/N per frames and tracks
    - [ ] Hyper-parameter tuning
    - [ ] ROC curve per parameter tuning
    - [ ] Turbine filter vs. no turbine filter ablation (which metric?)
  - [*] Min/max contour
  - [*] Turbine mask smoothing
  - [*] Expose h-params of each filter

- 10/26/2022
  - [] Push dirty repo into devel branch and prune master branch
  - [] Refactor metrics in vis.py into metrics.py
  - [] Write parameter search analysis jupyter notebook
  - [] Write tracklet code? (really want to)
  - [] Write test set video batching

# Video Labels

## Processed Label Structure

The structure of `label` when `video, label = next(dataloader)` is called:

```bash
{   'video_id': 23,
    'filename': '2010-09-08_183000_HF_S014.mp4',
    'video_length': 361,
    'video_shape': {
        'height': 1528,
        'width': 1024
    },
    'tracks': [
        {
            'track_id': 0,
            'label': 'target',
            'frames': [
                {
                    'frame': 236,
                    'box': ((658, 310), (696, 326)),
                    'occluded': 0,
                    'outside': 0,
                    'keyframe': 1
                },
                ...
            ]
        }
    ]
}
```

Structure of `preds` at the output of filtering pipeline

```bash
{   'video_id': 23,
    'filename': '2010-09-08_183000_HF_S014.mp4',
    'video_length': 361,
    'video_shape': {
        'height': 1528,
        'width': 1024
    },
    'tracks': [
        {
            'track_id': 0,
            'label': 'target',
            'frames': [
                {
                    'frame': 236,
                    'box': ((658, 310), (696, 326)),
                    'occluded': 0,
                    'outside': 0,
                    'keyframe': 1
                },
                ...
            ]
        }
    ]
}
```

# Video Notes

0: Easy
18: Hard, Small Fish
