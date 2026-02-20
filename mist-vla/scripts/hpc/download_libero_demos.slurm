#!/bin/bash
#SBATCH --job-name=dl_demos
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/download_demos_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/download_demos_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=shortq7

###############################################################################
#  Download LIBERO demonstration datasets (expert HDF5 files)
#  Needed for training DP + ACT baselines
###############################################################################

set -eo pipefail
export PYTHONUNBUFFERED=1

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/LIBERO

echo "================================================================"
echo "  Downloading LIBERO demonstration datasets"
echo "================================================================"

# Download all LIBERO benchmark datasets from HuggingFace
cd benchmark_scripts
python3 download_libero_datasets.py --datasets libero_spatial --use-huggingface
python3 download_libero_datasets.py --datasets libero_object --use-huggingface
python3 download_libero_datasets.py --datasets libero_goal --use-huggingface

echo ""
echo "=== Verifying downloads ==="
DATASET_DIR=$(python3 -c "import init_path; from libero.libero import get_libero_path; print(get_libero_path('datasets'))")
echo "Dataset dir: ${DATASET_DIR}"
find "${DATASET_DIR}" -name "*.hdf5" | head -30
echo ""
echo "Total HDF5 files: $(find "${DATASET_DIR}" -name "*.hdf5" | wc -l)"

echo ""
echo "================================================================"
echo "  DONE — LIBERO demos downloaded"
echo "================================================================"
