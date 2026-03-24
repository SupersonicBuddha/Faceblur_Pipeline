# sample_data/

This directory is for small test fixtures used by the test suite.

## What belongs here

- Tiny synthetic video clips (< 5 s, low resolution) for smoke tests.
- JSON/JSONL manifest fixtures.
- Small PNG/JPEG frames for unit tests.

## What does NOT belong here

- Real videos containing identifiable people.
- Videos larger than 5 MB.
- GCP credentials or configuration files.

## Generating synthetic test videos

Run from the project root:

```python
import cv2, numpy as np

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("sample_data/tiny_test.mp4", fourcc, 25.0, (320, 240))
for i in range(30):
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    writer.write(frame)
writer.release()
print("Written sample_data/tiny_test.mp4")
```
