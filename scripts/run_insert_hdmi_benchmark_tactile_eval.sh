#!/usr/bin/env bash
# =============================================================================
# UniVTAC Benchmark 评测 + 未来触觉预测对比
#
# 在 Isaac Sim 仿真中运行 insert_HDMI 任务，每个 action chunk (32步) 结束后：
#   - 对比模型预测的未来触觉 latent vs 仿真中实际采集的触觉
#   - 保存 GT/Pred 触觉图像对比图
#
# 用法:
#   cd /data1/cyy/UniVTAC
#   bash scripts/run_insert_hdmi_benchmark_tactile_eval.sh
#
# 可选环境变量:
#   GPU=0
#   TOTAL_NUM=5          # 评测 episode 数
#   DEPLOY_CONFIG=wla/deploy_tactile_encoder_predict_tactile_benchmark
# =============================================================================
set -euo pipefail

UNIVTAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${UNIVTAC_ROOT}"

GPU="${GPU:-0}"
TOTAL_NUM="${TOTAL_NUM:-5}"
TASK_NAME="${TASK_NAME:-insert_HDMI}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-wla/deploy_tactile_encoder_predict_tactile_benchmark}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate /data1/cyy/miniconda3/envs/UniVTAC
fi
if [[ -f "/data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh" ]]; then
  # shellcheck disable=SC1091
  source /data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "UniVTAC root:  ${UNIVTAC_ROOT}"
echo "Task:          ${TASK_NAME} / ${TASK_CONFIG}"
echo "Deploy config: ${DEPLOY_CONFIG}"
echo "GPU:           ${GPU}"
echo "Total episodes:${TOTAL_NUM}"

python scripts/eval_policy.py \
  "${TASK_NAME}" \
  "${TASK_CONFIG}" \
  "${DEPLOY_CONFIG}" \
  --total_num "${TOTAL_NUM}"

echo ""
echo "评测完成。触觉预测对比结果保存在:"
echo "  eval_result/wla/${TASK_NAME}/deploy_tactile_encoder_predict_tactile_benchmark/<timestamp>/tactile_pred/"
echo ""
echo "每个 action chunk 一个子目录:"
echo "  episode_XXX_step_YYYYY/tactile_image_compare.png"
echo "  episode_XXX_step_YYYYY/latent_metrics.png"
echo "  episode_XXX_step_YYYYY/tactile_contact_sheet.png"
echo "汇总:"
echo "  tactile_pred/summary.json"
