# UniAD

Unified autonomous driving inference and visualization pipeline on NuScenes. Uses ONNX Runtime or ailia to run BEV encoder, tracking, motion, occupancy, and planning heads; streams per-frame results to HDF5, then visualizes to images and video.

## Requirements

### Dependencies

Install the required Python packages:

```bash
pip3 install h5py
pip3 install casadi==3.6.7
pip3 install nuscenes-devkit==1.1.11
pip3 install pyquaternion==0.9.9
```

### Dataset Setup

Download the nuScenes V1.0 full dataset, CAN bus extension from [HERE](https://www.nuscenes.org/download), then place the dataset files under `data/nuscenes/`.  

Download nuScenes, CAN_bus and Map extensions:

```shell
cd uniad
mkdir data
cd uniad/data
mkdir nuscenes && cd nuscenes
# Download nuScenes V1.0 full dataset data directly to (or soft link to) uniad/data/nuscenes/
# Download CAN_bus directly to (or soft link to) uniad/data/nuscenes/
```

Prepare UniAD data info (Optional):

```shell
cd uniad/data
mkdir infos && cd infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl  # train_infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl  # val_infos
```

Directory structure:

```
uniad
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. Internet connection is required for the initial download.

For the sample image,
```bash
python3 uniad.py
```

This will process scenes `scene-0102` and `scene-0103` from `v1.0-trainval`, then visualize `scene-0103`.

### Workflow

The typical workflow consists of two steps:

1. Run model inference on specified scenes and save results to HDF5 file
2. Load HDF5 results and generate visualization images/videos

### Step 1: Run Inference

Run model inference and save results to HDF5:

```bash
python3 uniad.py --results_file all_frames.h5 \
	--data_root data/nuscenes/ \
	--version v1.0-trainval \
	--scenes scene-0102 --scenes scene-0103
```

Parameters:

- `--data_root`: Path to NuScenes dataset root (default: `data/nuscenes/`)
- `--version`: Dataset version: `v1.0-trainval`, `v1.0-test`, or `v1.0-mini` (default: `v1.0-trainval`)
- `--scenes`: Scene name(s) to process, repeatable (default: `scene-0102`, `scene-0103`)
- `--results_file`: Output HDF5 file path (default: `all_frames.h5`)

Process multiple scenes:

```bash
python3 uniad.py --results_file my_results.h5 \
	--scenes scene-0655 --scenes scene-0796 --scenes scene-0916
```

### Step 2: Visualize Results

Load HDF5 file and generate visualizations:

```bash
python3 uniad.py --visualize all_frames.h5 \
	--data_root data/nuscenes/ \
	--version v1.0-trainval \
	--vis_scenes scene-0103
```

This skips model inference and loads results directly from HDF5. Output includes:
- Visualization images (JPG) in `vis_output/` directory
- Video file as `output.avi`

Parameters:

- `--visualize`: Path to HDF5 file (required)
- `--vis_scenes`: Scene(s) to visualize, repeatable (default: all scenes in HDF5)
- `--data_root`: Path to NuScenes dataset root (default: `data/nuscenes/`)
- `--version`: Dataset version (default: `v1.0-trainval`)

Note: Frame rate is set to 4 fps and downsample factor is 2 (hardcoded).

Visualize specific scene:

```bash
python3 uniad.py --visualize all_frames.h5 --vis_scenes scene-0102
```

Visualize all scenes:

```bash
python3 uniad.py --visualize all_frames.h5
```

### Available Scenes

v1.0-trainval (850 scenes):
- Training: `scene-0001` to `scene-0700` (700 scenes)
- Validation: `scene-0701` to `scene-0850` (150 scenes)
- Recommended for testing: `scene-0102`, `scene-0103`, `scene-0655`, `scene-0796`, `scene-0916`

v1.0-test (150 scenes):
- `scene-0001` to `scene-0150`

v1.0-mini (10 scenes, for quick testing):
- `scene-0061`, `scene-0103`, `scene-0553`, `scene-0655`, `scene-0757`, `scene-0796`, `scene-0916`, `scene-1077`, `scene-1094`, `scene-1100`

## HDF5 Output Format

The HDF5 file contains per-frame results organized as sequential groups:

- Groups: `frame_000000`, `frame_000001`, ...
- Attributes: `token`, `scene_token`, `command`
- Datasets: `boxes_3d`, `scores_3d`, `labels_3d`, `track_ids`, `traj`, `traj_scores`, `sdc_boxes_3d`, `sdc_scores_3d`, `planning_traj`

## Reference

- [UniAD: Unified Driving](https://github.com/OpenDriveLab/UniAD)
