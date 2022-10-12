# TODOs

- 10/12/2022

  - [*] Auto-arranged windows across screen
  - [*] Debugged display bboxes on grayscale images (gray->color in vis) -> create new np.array that's 3-dim for each displayed video (was overwritting in place, but gray weren't right dims)
  - [*] Add out_format variable to filter classes
  - [*] Convert errode/dilate to cv2.opening (see docs)
    - [*] Erodes fish too much - also why are scale numbers not fully filtered by mean filter
    - [] Not temporal and insufficient; other options in findpeaks package

- [*] Made demos extensible to general filters

  - [*] demo: dft, mean, contour; run: dft ... refactored

- [] Ground filters in existent methods to reference and describe mathematically in paper
  - [] Background filter
  - [] Intensity filter?
- [] Improve filter output (lots of noise/speckle)
  - [] Implement speckle filter
- [] Implement and test additional filters
  - [] Finish implementing PIV filter

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
