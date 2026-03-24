# Face Blur Tool — V1

Automated, cloud-native batch pipeline that detects and blurs all real human
faces in videos stored in Google Cloud Storage.

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Architecture summary](#architecture-summary)
3. [Folder structure](#folder-structure)
4. [Dependency install](#dependency-install)
5. [Local run instructions](#local-run-instructions)
6. [Test instructions](#test-instructions)
7. [How the manifest works](#how-the-manifest-works)
8. [How output naming works](#how-output-naming-works)
9. [Limitations of V1](#limitations-of-v1)
10. [Colab setup instructions](#colab-setup-instructions)
11. [GCS auth options](#gcs-auth-options)
12. [Troubleshooting](#troubleshooting)
13. [Future improvements](#future-improvements)

---

## Project overview

Face Blur Tool reads videos from `gs://tbd-qa/for_fb`, detects and tracks
all visible human faces, blurs them with a padded elliptical Gaussian mask,
and writes the blurred output to `gs://tbd-qa/post_fb`.

**V1 design goals**

| Goal | Approach |
|------|----------|
| Zero privacy misses | Low confidence threshold, conservative track persistence, elliptical padding |
| Correctness | Two-pass decode (detect then write), interpolation across short gaps |
| Resumability | JSONL manifest; already-successful videos are skipped |
| Simplicity | No orchestration platform; runs from a single Colab notebook |
| Testability | Pure logic isolated from I/O; GCS interactions behind a thin adapter |

---

## Architecture summary

### Privacy risk analysis and mitigations

The highest risk in a face-blurring system is **missing a face** — a false
negative that leaks privacy.  Every design decision is biased toward recall:

1. **Low confidence threshold (0.35)** — errs toward detecting uncertain faces.
2. **Track persistence (12 frames)** — continues blurring after brief misses.
3. **Bbox padding (35%)** — the blur extends beyond the tight face bbox.
4. **Interpolation across gaps** — short detection misses between tracked frames
   are filled with linearly interpolated blur regions.
5. **Scene cut reset** — avoids tracker drift into wrong regions after a cut.
6. **No face detection on posters/screens** — out of scope; a detector trained
   on real faces will naturally have lower recall on 2-D printed faces.

### Module boundaries

```
GCS I/O  ─►  video_inventory  ─►  batch_runner
                                       │
                             ┌─────────▼──────────┐
                             │   process_video     │
                             │   ┌───────────────┐ │
                             │   │ probe         │ │
                             │   │ frame_reader  │ │
                             │   │ detector      │ │
                             │   │ tracker       │ │
                             │   │ smoother      │ │
                             │   │ interpolate   │ │
                             │   │ blur/mask     │ │
                             │   │ frame_writer  │ │
                             │   │ mux (ffmpeg)  │ │
                             │   └───────────────┘ │
                             └─────────────────────┘
                                       │
                                  manifest  ─►  GCS
```

### Detection strategy

We use **RetinaFace** (via `insightface` `buffalo_sc` model) as the primary
detector. It was chosen over alternatives because:

- Higher recall on tilted, partially occluded, and medium-distance faces
  compared to MTCNN and Haar cascades.
- Built-in 5-point landmarks (useful for future orientation-based padding).
- `insightface` runs on CPU without special hardware, scales to CUDA with
  zero code changes by switching ONNX execution providers.
- Well-maintained, Apache-2.0 licensed.

MediaPipe is included as a fallback for environments where `insightface`
cannot be installed (e.g. some ARM runtimes).

### Tracking strategy

We use **IoU-based matching with linear extrapolation** instead of a full
Kalman filter because:

- Simpler code, fewer tuneable parameters, easier to debug.
- For offline batch with detection every 4 frames, a lightweight motion model
  is sufficient.
- Hungarian assignment (`scipy.optimize.linear_sum_assignment`) gives optimal
  matching in O(n³) which is fine for the ≤ 10 simultaneous faces expected.
- A Kalman filter is a natural upgrade path if jitter becomes a problem.

---

## Folder structure

```
face_blur_tool/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── src/
│   ├── config.py            # FaceBlurConfig dataclass + env-var overrides
│   ├── constants.py         # Fixed design constants
│   ├── logging_utils.py     # Structured logging helpers
│   ├── gcs_io.py            # GCS adapter (thin wrapper around google-cloud-storage)
│   ├── manifest.py          # JSONL manifest read/write/skip logic
│   ├── video_inventory.py   # Discover eligible videos in GCS
│   ├── video_probe.py       # ffprobe metadata extraction
│   ├── frame_reader.py      # OpenCV streaming frame reader
│   ├── frame_writer.py      # OpenCV VideoWriter + ffmpeg audio mux
│   ├── ffmpeg_utils.py      # ffmpeg/ffprobe availability helpers
│   │
│   ├── detectors/
│   │   ├── base.py                  # FaceDetector ABC + Detection dataclass
│   │   ├── retinaface_detector.py   # insightface RetinaFace
│   │   ├── mediapipe_detector.py    # MediaPipe fallback
│   │   └── factory.py               # make_detector(config)
│   │
│   ├── tracking/
│   │   ├── types.py          # Track / TrackedFrame dataclasses
│   │   ├── tracker.py        # IoU + Hungarian multi-face tracker
│   │   ├── smoothing.py      # Rolling-average bbox smoother
│   │   ├── interpolation.py  # Linear bbox interpolation across gaps
│   │   └── scene_cut.py      # Histogram-delta scene cut detector
│   │
│   ├── blur/
│   │   ├── masks.py          # Ellipse mask generation
│   │   ├── blur_ops.py       # Gaussian blur operations
│   │   └── compositor.py     # Composite blurred faces onto original frame
│   │
│   ├── pipeline/
│   │   ├── process_video.py  # Single-video processing (download→detect→blur→upload)
│   │   ├── batch_runner.py   # Batch orchestration (discover→skip→retry→manifest)
│   │   ├── retry.py          # retry_once helper
│   │   └── result_types.py   # VideoResult dataclass
│   │
│   └── utils/
│       ├── paths.py          # Pure path manipulation (no I/O)
│       ├── timing.py         # Stopwatch / timed context manager
│       └── validation.py     # Input validation helpers
│
├── tests/
│   ├── test_config.py
│   ├── test_manifest.py
│   ├── test_video_inventory.py
│   ├── test_output_naming.py
│   ├── test_retry_logic.py
│   ├── test_scene_cut.py
│   ├── test_smoothing.py
│   ├── test_interpolation.py
│   ├── test_mask_generation.py
│   ├── test_blur_application.py
│   ├── test_process_video_smoke.py   # integration (marked)
│   └── test_batch_runner_smoke.py    # integration (marked)
│
├── notebooks/
│   └── FaceBlur_Batch_Colab.ipynb
│
└── sample_data/
    └── README.md
```

---

## Dependency install

### System dependencies

```bash
# Ubuntu / Debian (including Colab)
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1

# macOS (Homebrew)
brew install ffmpeg
```

### Python dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration:

```bash
pip install onnxruntime-gpu  # replaces onnxruntime
```

---

## Local run instructions

### 1. Set up credentials

```bash
# User auth (interactive)
gcloud auth application-default login

# Or set a service account key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
```

### 2. Run the batch processor

```python
from google.cloud import storage
from src.config import FaceBlurConfig
from src.pipeline.batch_runner import run_batch
from src.logging_utils import configure_root_logger

configure_root_logger(level="INFO")

client = storage.Client(project="YOUR_PROJECT_ID")
config = FaceBlurConfig()

summary = run_batch(client, config)
summary.print_summary()
```

### 3. Dry run (discover videos without processing)

```python
run_batch(client, config, dry_run=True)
```

---

## Test instructions

### Run all unit tests (no GCS / GPU required)

```bash
cd face_blur_tool
pytest tests/ -m "not integration" -v
```

### Run integration / smoke tests (require OpenCV + ffmpeg locally)

```bash
pytest tests/ -v
```

### Run a specific test file

```bash
pytest tests/test_manifest.py -v
```

---

## How the manifest works

The manifest is stored as **JSONL** at:

```
gs://tbd-qa/post_fb/processed_manifest.jsonl
```

Each line is a JSON record:

```json
{
  "source_gcs_path": "gs://tbd-qa/for_fb/session1/video.mp4",
  "output_gcs_path": "gs://tbd-qa/post_fb/session1/video_blurred.mp4",
  "status": "success",
  "processed_at": "2024-05-01T14:23:00+00:00",
  "retry_count": 0,
  "error_message": null,
  "model_version": "v1.0.0",
  "source_extension": ".mp4",
  "duration_sec": 120.5,
  "width": 1920,
  "height": 1080,
  "fps": 29.97
}
```

**Skip logic**: At batch start, the manifest is loaded into memory.  Any
video whose `source_gcs_path` appears with `status=success` is skipped.
Failed or missing videos are processed.

**Uniqueness key**: `source_gcs_path` (full GCS URI).  If the same video is
recorded twice (e.g. retry), the later record replaces the earlier one.

**GCS write**: After each video, the full manifest is serialised and
re-uploaded.  (GCS does not support true append; for V1 batch sizes the
round-trip is acceptable.)

---

## How output naming works

Given:

```
input:  gs://tbd-qa/for_fb/foo/bar/video1.mp4
```

Output:

```
output: gs://tbd-qa/post_fb/foo/bar/video1_blurred.mp4
```

Rules:
- Relative folder structure under the input prefix is preserved.
- `_blurred` is inserted before the file extension.
- Extension is preserved unchanged (case-sensitive).
- Original resolution and FPS are preserved.

---

## Limitations of V1

| Limitation | Notes |
|------------|-------|
| Single-threaded batch | Videos are processed sequentially. Parallelism is a V2 item. |
| No face on posters/screens | Out of scope per spec. A real-face detector naturally has low recall on printed faces. |
| No tiny-face optimisation | Faces smaller than `min_face_size` (30 px) are ignored. |
| No person re-ID across long gaps | Tracker resets on scene cuts; long-duration re-ID is out of scope. |
| No identity whitelisting | No approved-face logic. All faces are blurred. |
| Manifest write is not atomic | Re-upload on every video. A crash mid-write could leave a partial manifest. Mitigated by idempotency (re-running re-processes anything not marked success). |
| Audio re-encoded to AAC | Original audio codec is not preserved (mux step). |
| Two-pass decode | Each video is decoded twice (once for detection, once for writing). A streaming single-pass approach is a performance improvement for V2. |

---

## Colab setup instructions

### Prerequisites

1. A Google account with access to the `tbd-qa` GCS bucket.
2. Your account needs these IAM roles on the bucket (or project):
   - `roles/storage.objectViewer` on `tbd-qa/for_fb` (to read input)
   - `roles/storage.objectCreator` on `tbd-qa/post_fb` (to write output + manifest)

### Step-by-step

#### 1. Get the code into Colab

**Option A — GitHub clone** (recommended):
```python
!git clone https://github.com/YOUR_ORG/face_blur_tool.git
%cd face_blur_tool
```

**Option B — Upload a zip**:
1. Zip the `face_blur_tool/` directory on your machine.
2. In Colab, click the folder icon (left sidebar) → Upload icon.
3. Upload `face_blur_tool.zip`.
4. Unzip:
   ```python
   !unzip face_blur_tool.zip -d /content/
   ```

#### 2. Install system dependencies

```python
!apt-get update -qq && apt-get install -y -qq ffmpeg libgl1
```

#### 3. Install Python dependencies

```python
!pip install -r /content/face_blur_tool/requirements.txt
```

#### 4. Add the project to the Python path

```python
import sys
sys.path.insert(0, "/content/face_blur_tool")
```

#### 5. Authenticate to GCP

```python
from google.colab import auth
auth.authenticate_user()
```

Then set your project:
```python
!gcloud config set project YOUR_PROJECT_ID
```

#### 6. Confirm bucket access

```python
from google.cloud import storage
client = storage.Client()
blobs = list(client.list_blobs("tbd-qa", prefix="for_fb", max_results=5))
print([b.name for b in blobs])
```

#### 7. Run the notebook

Open `notebooks/FaceBlur_Batch_Colab.ipynb` and run cells in order.

#### 8. Re-run safely

Re-running the notebook is safe.  The manifest skip logic ensures already-
successful videos are not reprocessed.

To force reprocessing of a specific video, remove its line from the manifest:

```python
# Download the manifest, remove the line, re-upload
from src import gcs_io
lines = gcs_io.read_jsonl_blob(client, "tbd-qa", "post_fb/processed_manifest.jsonl")
filtered = [l for l in lines if "video_to_reprocess.mp4" not in l]
gcs_io.upload_text_as_blob(
    client, "tbd-qa", "post_fb/processed_manifest.jsonl",
    "\n".join(filtered) + "\n"
)
```

---

## GCS auth options

| Method | When to use |
|--------|-------------|
| `google.colab.auth.authenticate_user()` | Interactive Colab sessions with a personal Google account |
| `GOOGLE_APPLICATION_CREDENTIALS` env var | Service account JSON key; CI/CD; non-interactive environments |
| Workload Identity (GKE/Cloud Run) | When running on GCP compute — no key file needed |
| `gcloud auth application-default login` | Local development |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Add the project root to `sys.path` (see Step 4 above) |
| `ffprobe not found` | Install ffmpeg: `!apt-get install -y ffmpeg` |
| `DefaultCredentialsError` | Re-authenticate with `auth.authenticate_user()` |
| `403 Forbidden` on GCS read | Check IAM; your account needs `storage.objectViewer` on `tbd-qa` |
| `403 Forbidden` on GCS write | Your account needs `storage.objectCreator` on `tbd-qa` |
| `insightface` install fails on macOS ARM | Use `detector_type="mediapipe"` as a fallback |
| Video marked `failed` in manifest | Check `error_message` field; fix the issue; delete the manifest line to retry |
| Output video has no audio | Confirm `preserve_audio=True` in config and that the source has audio |
| Runtime disconnects mid-batch | Re-run — already-successful videos are skipped |
| Very slow on CPU | Use a Colab GPU runtime; or switch to `mediapipe` for faster (lower recall) detection |

---

## Future improvements

### Short-term (V1.1)

- **Single-pass processing**: Stream frames to a rolling buffer, detect in
  parallel, and write frames immediately to avoid double-decoding.
- **GPU ONNX providers**: Pass `['CUDAExecutionProvider', 'CPUExecutionProvider']`
  to `RetinaFaceDetector` when a CUDA device is available.
- **Parallel batch**: Process multiple videos concurrently with
  `concurrent.futures.ThreadPoolExecutor` (GCS I/O is the bottleneck, not CPU).

### Medium-term (V2)

- **Containerise**: Wrap in a Docker image; run as a Cloud Run Job or GKE
  batch workload instead of Colab.
- **Streaming manifest**: Use Firestore or BigQuery instead of JSONL for
  concurrent-safe atomic updates.
- **Per-track interpolation**: Use `tracking/interpolation.py`'s `fill_gaps`
  on a per-track basis rather than the merged per-frame approach.
- **Orientation-aware padding**: Use face landmarks to compute a rotated
  ellipse aligned with the face orientation.

### Long-term (V3+)

- **Human review UI**: Flag low-confidence detections for manual review.
- **Live streaming support**: Adapt the pipeline to process RTSP/HLS streams.
- **Identity whitelisting**: Allow pre-approved faces to pass through unblurred.
