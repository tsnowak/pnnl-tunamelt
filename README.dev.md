# TODOs

- [] Modify all filters to only use (HSV) value channel [NxWxH or NxWxHx1]
- [] Modify filters to be grounded in prior work/common implementations as much as possible
- [] After modification identify if speckle filter is needed (Wavelet?)
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
