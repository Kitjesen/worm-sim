# Worm V6 Representative Simulation Videos

Complete: `true`

| Terrain | Policy | gait_blend | Status | Video | Metrics |
| --- | --- | ---: | --- | --- | --- |
| flat | random | 0.750 | ok | record/v6/videos/eval_flat_random.mp4 | record/v6/paper_results/video_eval_flat_random.json |
| sand | random | 0.500 | ok | record/v6/videos/eval_sand_random.mp4 | record/v6/paper_results/video_eval_sand_random.json |
| slope | random | 0.500 | ok | record/v6/videos/eval_slope_random.mp4 | record/v6/paper_results/video_eval_slope_random.json |

## Reproduce

### flat

```powershell
C:\Users\99563\miniconda3\python.exe D:\inovxio\thirdparty\simulation\worm_project\src\v6\eval_v6.py --terrain flat --gait-mode random --gait-blend 0.750 --episodes 1 --time 6.0 --seed 2200 --json-out D:\inovxio\thirdparty\simulation\worm_project\record\v6\paper_results\video_eval_flat_random.json --video
```

### sand

```powershell
C:\Users\99563\miniconda3\python.exe D:\inovxio\thirdparty\simulation\worm_project\src\v6\eval_v6.py --terrain sand --gait-mode random --gait-blend 0.500 --episodes 1 --time 6.0 --seed 2200 --json-out D:\inovxio\thirdparty\simulation\worm_project\record\v6\paper_results\video_eval_sand_random.json --video
```

### slope

```powershell
C:\Users\99563\miniconda3\python.exe D:\inovxio\thirdparty\simulation\worm_project\src\v6\eval_v6.py --terrain slope --gait-mode random --gait-blend 0.500 --episodes 1 --time 6.0 --seed 2200 --json-out D:\inovxio\thirdparty\simulation\worm_project\record\v6\paper_results\video_eval_slope_random.json --video
```
