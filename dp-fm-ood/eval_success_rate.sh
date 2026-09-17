#!/bin/bash
# Batch closed-loop success-rate evaluation across checkpoints.
#
# This is the number the replay check CANNOT give you. dp-fm-ood/replay_check.py measures
# open-loop action-prediction accuracy (R^2) against recorded training actions -- nothing is
# executed and no blocks move. Success rate requires actually driving the robot in Isaac and
# checking env_cfg.terminations.success, which is what this does.
#
# For each checkpoint it:
#   1. starts multitask_dit_server.py holding that checkpoint (CPU by default, see SERVER_DEVICE)
#   2. waits for its "listening on ..." line
#   3. runs run_policy_fm.py for N rollouts, --headless --enable_cameras (GPU, ~6-7 GB).
#      --enable_cameras is REQUIRED: the policy conditions on table_cam/wrist_cam, and without
#      it Isaac never instantiates the camera sensors, so the observation dict has no images.
#   4. parses the "Success rate:" line
#   5. kills the server and moves on
#
# RESOURCES -- read before launching. Isaac with cameras needs ~6.6 GB. The policy server runs
# on the CPU by default (SERVER_DEVICE=cpu), which costs ~1.5 s per 24-step cycle and keeps its
# ~2 GB off the card, so only Isaac competes with whatever else is running. A concurrent
# training run holds ~22-25 GB of a 32.6 GB card, which leaves this marginal: the preflight
# refuses to start below MIN_FREE_MIB. Do not lower that casually -- an OOM here will most
# likely kill the TRAINING process, not this one.
#
# The task string must match what the checkpoint trained on. The N=50/100 datasets were
# corrected from "adn" to "and" on 2026-09-02; checkpoints trained BEFORE that saw "adn".
# Measured impact was nil (within seed noise), but override TASK_INSTRUCTION if you want to be
# faithful to an older checkpoint.
#
# Usage:
#   ./eval_success_rate.sh CKPT_DIR [CKPT_DIR ...]
#   N_ROLLOUTS=20 ./eval_success_rate.sh ../outputs/run_0902_norm/n100_k10_s100k_hz32_L6/checkpoints/*/
#   DRY=true ./eval_success_rate.sh <dir>          # print what would run
#
# A CKPT_DIR is either a .../checkpoints/<step>/ dir or its pretrained_model/ subdir.
set -u
cd "$(dirname "$0")"

TASK=${TASK:-Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0}
N_ROLLOUTS=${N_ROLLOUTS:-20}
HORIZON=${HORIZON:-500}
SEED=${SEED:-42}
PORT=${PORT:-5555}
AUTHKEY=${AUTHKEY:-mdit-ipc}
# Run the policy server on the CPU to leave the GPU entirely to Isaac. Measured cost: a chunk
# generation (100 Euler steps) is ~1.5 s and the other 23 steps of the cycle just pop a queued
# action, so a 500-step rollout spends ~30 s in the policy -- Isaac's stepping dominates either
# way. Worth ~2 GB of VRAM when a training run is holding the card.
SERVER_DEVICE=${SERVER_DEVICE:-cpu}
# Optional inference-time override of how much of each chunk is executed before replanning.
# Empty = use the checkpoint's trained value (24 = 0.8 s at 30 Hz). This needs NO retraining,
# so the reactivity question can be swept on existing checkpoints: N_ACTION_STEPS=8 is 0.27 s.
N_ACTION_STEPS=${N_ACTION_STEPS:-}
# Isaac with cameras needs ~6.6 GB. With the server on the CPU that is all we need; on the GPU
# add ~2 GB for the policy.
if [ "${SERVER_DEVICE}" = "cpu" ]; then
    MIN_FREE_MIB=${MIN_FREE_MIB:-7000}
else
    MIN_FREE_MIB=${MIN_FREE_MIB:-9000}
fi
SERVER_TIMEOUT=${SERVER_TIMEOUT:-300}
# The two halves need DIFFERENT conda envs, which is the whole point of the client/server split:
#   leisaac                     -> has isaaclab, lerobot 0.4.2   -> runs run_policy_fm.py
#   lerobot_0.6.1_multitask_dit -> has lerobot 0.6.1, no isaaclab -> runs multitask_dit_server.py
# 0.4.2 has no lerobot.policies.make_pre_post_processors, so starting the server in the Isaac env
# fails with an ImportError. Run this script from the Isaac env; the server is launched into
# SERVER_CONDA_ENV in a subshell.
SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-lerobot_0.6.1_multitask_dit}
CONDA_SH=${CONDA_SH:-/home/wisc-rt2-trimanual/miniconda3/etc/profile.d/conda.sh}
TASK_INSTRUCTION=${TASK_INSTRUCTION:-"grab red block and stack on top of blue block, then grab green block and stack on top of red block"}
OUT_CSV=${OUT_CSV:-./outputs/success_rates.csv}
DRY=${DRY:-false}

if [ "$#" -eq 0 ]; then
    echo "usage: $0 CKPT_DIR [CKPT_DIR ...]"; exit 2
fi

mkdir -p "$(dirname "${OUT_CSV}")" ./outputs/eval_logs
[ -f "${OUT_CSV}" ] || echo "checkpoint,rollouts,successes,success_rate,timestamp" > "${OUT_CSV}"

free_mib() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) \
                    - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) )); }

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  [cleanup] killing server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
# Without this an interrupted run leaves a server holding the port AND ~2 GB of VRAM, and the
# next checkpoint fails its handshake with a confusing address-in-use error.
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

# Fail fast on the env split rather than after a 2-minute Isaac boot: 0.4.2 lacks this symbol.
if ! bash -c "source '${CONDA_SH}' && conda activate '${SERVER_CONDA_ENV}' && \
     python -c 'from lerobot.policies import make_pre_post_processors'" 2>/dev/null; then
    echo "ABORT: conda env '${SERVER_CONDA_ENV}' cannot import make_pre_post_processors."
    echo "       That symbol needs lerobot >= 0.6. Set SERVER_CONDA_ENV to the env that has it."
    exit 1
fi

FREE=$(free_mib)
echo "=================================================================="
echo "  SUCCESS-RATE EVAL   task=${TASK}"
echo "  ${N_ROLLOUTS} rollouts x ${#} checkpoint(s), horizon ${HORIZON}, seed ${SEED}"
echo "  server: ${SERVER_CONDA_ENV} on ${SERVER_DEVICE}   |   VRAM free: ${FREE} MiB (need >= ${MIN_FREE_MIB})"
echo "=================================================================="
if [ "${FREE}" -lt "${MIN_FREE_MIB}" ]; then
    echo "ABORT: only ${FREE} MiB free; need ${MIN_FREE_MIB} MiB (Isaac ~6.6 GB, server on ${SERVER_DEVICE})."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "Set MIN_FREE_MIB lower to override, but an OOM will most likely kill the OTHER job."
    exit 1
fi

SEEN=""
for RAW in "$@"; do
    CKPT=${RAW%/}
    [ -d "${CKPT}/pretrained_model" ] && CKPT="${CKPT}/pretrained_model"
    if [ ! -f "${CKPT}/config.json" ]; then
        echo ">>> SKIP ${RAW}: no config.json (not a pretrained_model dir)"; continue
    fi
    # checkpoints/last is a symlink to the newest numbered dir, so a glob passes the same
    # checkpoint twice. Dedupe on the resolved path rather than the argument.
    REAL=$(readlink -f "${CKPT}")
    case " ${SEEN} " in
        *" ${REAL} "*) echo ">>> SKIP ${RAW}: same checkpoint as one already evaluated"; continue ;;
    esac
    SEEN="${SEEN} ${REAL}"
    # .../<run>/checkpoints/<step>/pretrained_model -> "<run>@<step>"
    STEP=$(basename "$(dirname "${CKPT}")")
    RUN=$(basename "$(dirname "$(dirname "$(dirname "${CKPT}")")")")
    TAG="${RUN}@${STEP}"
    [ -n "${N_ACTION_STEPS}" ] && TAG="${TAG}_nas${N_ACTION_STEPS}"
    LOG=./outputs/eval_logs/${TAG//\//_}.log

    echo ""
    echo "=================================================================="
    echo "  ${TAG}"
    echo "  -> ${LOG}"
    echo "=================================================================="

    if [ "${DRY}" = "true" ]; then
        echo "  would start: [${SERVER_CONDA_ENV}] python multitask_dit_server.py --checkpoint ${CKPT} --port ${PORT} --device ${SERVER_DEVICE}"
        echo "  would run  : python run_policy_fm.py --task ${TASK} --fm_checkpoint ${CKPT} --num_rollouts ${N_ROLLOUTS} --headless --enable_cameras"
        continue
    fi

    SERVER_LOG=${LOG%.log}.server.log
    NAS_ARG=""
    [ -n "${N_ACTION_STEPS}" ] && NAS_ARG="--n_action_steps ${N_ACTION_STEPS}"
    # python -u so the "listening on" banner is unbuffered and the poll below sees it promptly.
    bash -c "source '${CONDA_SH}' && conda activate '${SERVER_CONDA_ENV}' && \
        exec python -u multitask_dit_server.py --checkpoint '${CKPT}' --port '${PORT}' \
        --device '${SERVER_DEVICE}' --authkey '${AUTHKEY}' ${NAS_ARG}" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    # Poll for the listener banner rather than sleeping a fixed amount: CLIP init varies.
    READY=false
    for _ in $(seq 1 "${SERVER_TIMEOUT}"); do
        if grep -q "listening on" "${SERVER_LOG}" 2>/dev/null; then READY=true; break; fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then break; fi
        sleep 1
    done
    if [ "${READY}" != "true" ]; then
        echo "  ERROR: server never became ready. Tail of ${SERVER_LOG}:"
        tail -5 "${SERVER_LOG}"
        cleanup; SERVER_PID=""; continue
    fi
    echo "  server up (pid ${SERVER_PID}), starting ${N_ROLLOUTS} rollouts..."

    python run_policy_fm.py \
        --task "${TASK}" \
        --fm_backbone multitask_dit \
        --fm_checkpoint "${CKPT}" \
        --mdit_server_port "${PORT}" \
        --mdit_server_authkey "${AUTHKEY}" \
        --task_instruction "${TASK_INSTRUCTION}" \
        --num_rollouts "${N_ROLLOUTS}" \
        --horizon "${HORIZON}" \
        --seed "${SEED}" \
        --headless \
        --enable_cameras 2>&1 | tee "${LOG}"
    RC=${PIPESTATUS[0]}

    cleanup; SERVER_PID=""

    if [ "${RC}" -ne 0 ]; then
        echo "  ${TAG} FAILED (exit ${RC}) -- see ${LOG}"
        continue
    fi

    # "Successful trials: 7, out of 20 trials"
    LINE=$(grep -a "Successful trials:" "${LOG}" | tail -1)
    SUCC=$(echo "${LINE}" | grep -oE "trials: [0-9]+" | grep -oE "[0-9]+")
    TOT=$(echo "${LINE}"  | grep -oE "of [0-9]+ trials" | grep -oE "[0-9]+")
    if [ -z "${SUCC}" ] || [ -z "${TOT}" ] || [ "${TOT}" = "0" ]; then
        echo "  ${TAG}: could not parse a success line from ${LOG}"; continue
    fi
    RATE=$(awk "BEGIN{printf \"%.3f\", ${SUCC}/${TOT}}")
    echo "${TAG},${TOT},${SUCC},${RATE},$(date '+%Y-%m-%d %H:%M:%S')" >> "${OUT_CSV}"
    printf "  >>> %s : %s/%s = %s\n" "${TAG}" "${SUCC}" "${TOT}" "${RATE}"
done

echo ""
echo "=================================================================="
echo "  RESULTS  (${OUT_CSV})"
echo "=================================================================="
column -t -s, "${OUT_CSV}"
