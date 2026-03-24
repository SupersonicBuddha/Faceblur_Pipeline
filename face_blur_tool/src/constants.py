"""
Project-wide constants.  Values here are NOT user-configurable;
they represent fixed design decisions.
"""

# Blurred-output filename suffix (inserted before the extension)
BLURRED_SUFFIX = "_blurred"

# Manifest record status values
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# JSONL manifest filename (relative within output prefix)
MANIFEST_FILENAME = "processed_manifest.jsonl"

# Maximum number of frames held in the per-video frame buffer
# (kept small to avoid OOM on long videos)
FRAME_BUFFER_SIZE = 32

# OpenCV colour channel order used throughout (BGR)
CV2_CHANNEL_ORDER = "BGR"

# Minimum number of frames required to consider scene-cut detection active
SCENE_CUT_MIN_FRAMES = 2

# Track ID counter reset sentinel
TRACK_ID_RESET_SENTINEL = -1

# Supported container → ffmpeg demuxer map (for validation)
FFMPEG_DEMUXER = {
    ".mp4": "mp4",
    ".mov": "mov,mp4,m4a,3gp,3g2,mj2",
}
