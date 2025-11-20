#!/bin/bash
#SBATCH --job-name=tomopanda-pick_2d_projection
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00




# Usage:
#   bash run_force_angpix.sh a.txt 1.34 8
#   参数1：包含 mrc 路径的 txt 文件
#   参数2：pixel size (Å)
#   参数3：并行线程数 (N)

LIST_FILE="subtomos.txt"
PIXEL_SIZE="6.72"
NPROC="16"

export PIXEL_SIZE

parallel -j ${NPROC} '
    INPUT_MRC="{}";
    relion_image_handler \
        --i "${INPUT_MRC}" \
        --o "${INPUT_MRC}" \
        --force_header_angpix ${PIXEL_SIZE}
' :::: "${LIST_FILE}"