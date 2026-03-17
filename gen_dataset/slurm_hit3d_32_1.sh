#!/usr/bin/bash
#SBATCH --job-name=hit3d
#SBATCH --output=slogs/hit3d_%A_%a.out
#SBATCH --error=slogs/hit3d_%A_%a.out
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1

# Launch with
# sbatch --array=0-19 gen_dataset/slurm_hit3d_32_1.sh

set -e

DATA_ROOT=/local/disk/atoshev/dataset_hit
seed=$SLURM_ARRAY_TASK_ID

echo "Run with seed = $seed"

### Simulate trajectories
python gen_dataset/jaxsph_main.py config=jaxsph_cases/rlx.yaml seed=$seed case.dim=3 \
    case.dx=0.1963495408 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 \
    io.data_path=$DATA_ROOT/data_relaxed/ io.print_props=['Ekin','u_max','rho_max']

python main.py config=configs/hit32_1.yaml mode=combined seed=$seed \
    com.state_0_path=$DATA_ROOT/data_relaxed/rlx_3_0.1963495408_${seed}.h5 \
    com.dst_path=$DATA_ROOT/raw/3D_HIT_32768_20kevery1/traj_${seed}

### After trajectory simulation:

# 1. Analyze created trajectories
# python scripts/inspect_hit3d_com.py

# 2. Create dataset from trajectories
# python gen_dataset/gen_dataset.py --split=2_1_1 \
#     --skip_first_n_frames=0 --slice_every_nth_frame=1 \
#     --src_dir=$DATA_ROOT/raw/3D_HIT_32768_20kevery1/ \
#     --dst_dir=$DATA_ROOT/datasets/3D_HIT_32768_20kevery1/
