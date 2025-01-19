#!/usr/bin/bash
# nohup ./scripts/dataset_tgv.sh 2>&1 &

DATA_ROOT=../dataset_kolm

for seed in {0..19}
do
    echo "Run with seed = $seed"

    python jaxsph_main.py config=jaxsph_cases/rlx.yaml seed=$seed case.dim=2 case.dx=0.0981747704 \
        case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/data_relaxed/

    python ../main.py config=../configs/kolmogorov64.yaml mode=combined seed=$seed \
        int.state_0_path=$DATA_ROOT/data_relaxed/rlx_2_0.0981747704_${seed}.h5 \
        com.dst_path=$DATA_ROOT/raw/2D_KOLM_4096_20kevery10/traj_${seed}
done
python gen_dataset.py --split=2_1_1 --skip_first_n_frames=100 \
    --src_dir=$DATA_ROOT/raw/2D_KOLM_4096_20kevery10/ \
    --dst_dir=$DATA_ROOT/datasets/2D_KOLM_4096_20kevery10/
