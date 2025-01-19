#!/usr/bin/bash
#SBATCH --job-name=kolm2d
#SBATCH --output=jobs/kolm2d_%A_%a.out
#SBATCH --error=jobs/kolm2d_%A_%a.err
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

DATA_ROOT=../dataset_kolm
seed=$SLURM_ARRAY_TASK_ID

echo "Run with seed = $seed"

python jaxsph_main.py config=jaxsph_cases/rlx.yaml seed=$seed case.dim=2 case.dx=0.0981747704 \
    case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/data_relaxed/

python ../main.py config=../configs/kolmogorov64.yaml mode=combined seed=$seed \
    int.state_0_path=$DATA_ROOT/data_relaxed/rlx_2_0.0981747704_${seed}.h5 \
    com.dst_path=$DATA_ROOT/raw/2D_KOLM_4096_20kevery10/traj_${seed}