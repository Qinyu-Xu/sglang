#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_experiment.sh" \
  \
  `# *** IMPORTANT ***` \
  --model-path            "Qwen/Qwen3-8B" \
  --utilizations          "1.4" \
  --long-fraction         0.2 \
  --clip-max-new-tokens   4096 \
  --clip-max-new-tokens-chat 128 \
  \
  `# server` \
  --port                  39942 \
  --mem-fraction          0.5 \
  --dtype                 auto \
  --schedule-conservativeness 1.0 \
  --server-ready-timeout  300 \
  --skip-long-on-no-token \
  --max-concurrent-chat   50 \
  \
  `# sweep` \
  --total-requests        500 \
  --seed                  42 \
  --request-timeout       900 \
  --cooldown              15 \
  --chat-priority         0
