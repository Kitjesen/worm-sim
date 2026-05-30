# Record Directory Layout

`record/` is organized by artifact purpose, not by code version.

## Current Entry Points

- `record/VIDEO_INDEX.md` is the first place to find videos.
- `record/current/` contains current paper-facing or demo-facing artifacts.
- `record/current/flat_omni_v29_hd/` contains the latest HD flat six-direction
  V29 videos, metrics, trajectories, and thumbnails.

## Legacy Archives

- `record/v6/` is now treated as a legacy V6 experiment archive. It still
  contains older videos, scans, paper-result drafts, and historical diagnostics.
- New current videos should not be added under `record/v6/`; add them under
  `record/current/<descriptive-name>/` and update `record/VIDEO_INDEX.md`.

This split avoids a misleading path such as `record/v6` for the newest
paper-facing media while keeping old links and historical artifacts available.
