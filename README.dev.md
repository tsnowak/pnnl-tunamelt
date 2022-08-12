# TODOs

- create metrics for comparing filter output to labels
- create standard viz from dataloader
- viz data and labels to verify integrity
- improve/test additional filters

# Video Labels

## Processed Label Structure

The structure of `label` when `video, label = next(dataloader)` is called:

``` bash
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
