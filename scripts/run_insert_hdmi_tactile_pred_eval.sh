#!/usr/bin/env bash
# =============================================================================
# UniVTAC: 评估 insert_hdmi tactile-encoder + predict_tactile 模型的未来触觉预测
#
# 使用配置:
#   Mantis: configs/insert_hdmi_tactile_encoder_lerobot_image_action_predict_tactile.yaml
#   Deploy: policy/wla/deploy_tactile_encoder_predict_tactile.yml
#
# 用法:
#   cd /data1/cyy/UniVTAC
#   bash scripts/run_insert_hdmi_tactile_pred_eval.sh
#
# 可选环境变量:
#   DEPLOY_CONFIG=.../deploy_tactile_encoder_predict_tactile.yml
#   OUTPUT_DIR=.../eval_result/wla/insert_HDMI/tactile_pred_vis
#   DATASET_ROOT=/path/to/insert_hdmi_tactile_lerobot_hw_270_480
#   SAMPLE_INDICES=0,50,100,200,500
# =============================================================================
set -euo pipefail

UNIVTAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${UNIVTAC_ROOT}"

DEPLOY_CONFIG="${DEPLOY_CONFIG:-${UNIVTAC_ROOT}/policy/wla/deploy_tactile_encoder_predict_tactile.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-${UNIVTAC_ROOT}/eval_result/wla/insert_HDMI/tactile_pred_vis}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/netdata/Team/Personal/chenyiyang/datasets/adk111/insert_hdmi_tactile_lerobot_hw_270_480}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,50,100,200,500}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
TIMESTEPS="${TIMESTEPS:-0,7,15,23,31}"

# 激活 UniVTAC 环境 (与 activate_env.sh 一致)
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate /data1/cyy/miniconda3/envs/UniVTAC
fi
if [[ -f "/data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh" ]]; then
  # shellcheck disable=SC1091
  source /data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh
fi

echo "UniVTAC root:   ${UNIVTAC_ROOT}"
echo "Deploy config:  ${DEPLOY_CONFIG}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Dataset root:   ${DATASET_ROOT}"
echo "Python:         $(which python)"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CMD=(
  python "${UNIVTAC_ROOT}/scripts/eval_pred_tactile_univtac.py"
  --deploy_config "${DEPLOY_CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_root_dir "${DATASET_ROOT}"
  --sample_indices "${SAMPLE_INDICES}"
  --num_samples "${NUM_SAMPLES}"
  --timesteps "${TIMESTEPS}"
  --inspect_decoder
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "完成。查看结果:"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/sample_XXXXXX/tactile_image_compare.png"
echo "  ${OUTPUT_DIR}/sample_XXXXXX/tactile_contact_sheet.png"
echo "  ${OUTPUT_DIR}/sample_XXXXXX/latent_metrics.png"
echo "  ${OUTPUT_DIR}/sample_XXXXXX/latent_heatmap.png"
