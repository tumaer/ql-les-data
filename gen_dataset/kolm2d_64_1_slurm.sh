#!/usr/bin/bash
#SBATCH --job-name=kolm2d
#SBATCH --output=slogs/kolm2d_%A_%a.out
#SBATCH --error=slogs/kolm2d_%A_%a.out
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

# Launch with
# cd gen_dataset && sbatch --array=0-19 kolm2d_64_1_slurm.sh

set -e

DATA_ROOT=/local/disk/atoshev/dataset_kolm
seed=$SLURM_ARRAY_TASK_ID

echo "Run with seed = $seed"

# # Simulate trajectories
# python jaxsph_main.py config=jaxsph_cases/rlx.yaml seed=$seed case.dim=2 case.dx=0.0981747704 \
#     case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/data_relaxed/

# python ../main.py config=../configs/kolmogorov64_1.yaml mode=combined seed=$seed \
#     int.state_0_path=$DATA_ROOT/data_relaxed/rlx_2_0.0981747704_${seed}.h5 \
#     com.dst_path=$DATA_ROOT/raw/2D_KOLM_4096_140kevery1/traj_${seed}

# # Create dataset
# python gen_dataset.py --split=2_1_1 --skip_first_n_frames=4500 --slice_every_nth_frame=1 \
#     --src_dir=$DATA_ROOT/raw/2D_KOLM_4096_140kevery1/ \
#     --dst_dir=$DATA_ROOT/datasets/2D_KOLM_4096_140kevery1/

# python gen_dataset.py --split=2_1_1 --skip_first_n_frames=4500 \
#     --dst_dir=$DATA_ROOT/datasets/2D_KOLM_4096_140kevery1 \
#     --stats_every_nth=10 --only_stats