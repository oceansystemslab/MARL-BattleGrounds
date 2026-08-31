#!/usr/bin/env bash

# Shared bounded-process runner for local validation scripts.
#
# This file deliberately owns no test inventory. Callers provide labels and
# argv arrays through marl_validation_start; commands are never reparsed with
# eval.

declare -Ag MARL_VALIDATION_TASK_BY_PID=()
declare -Ag MARL_VALIDATION_LABEL_BY_TASK=()
declare -Ag MARL_VALIDATION_LOG_BY_TASK=()
declare -Ag MARL_VALIDATION_RESULT_BY_TASK=()
declare -Ag MARL_VALIDATION_STATUS_BY_TASK=()
declare -Ag MARL_VALIDATION_DURATION_BY_TASK=()
declare -ag MARL_VALIDATION_TASK_ORDER=()

MARL_VALIDATION_MAX_JOBS=0
MARL_VALIDATION_NEXT_TASK=0
MARL_VALIDATION_FAILURES=0
MARL_VALIDATION_LOG_DIR=""
MARL_VALIDATION_TRAPS_INSTALLED=0

marl_validation_terminate_process_tree() {
  local parent_pid="$1"
  local child_pid=""

  while IFS= read -r child_pid; do
    if [[ -n "${child_pid}" ]]; then
      marl_validation_terminate_process_tree "${child_pid}"
    fi
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
  kill -TERM "${parent_pid}" 2>/dev/null || true
}

marl_validation_handle_signal() {
  local signal_name="$1"
  local exit_status=143

  if [[ "${signal_name}" == INT ]]; then
    exit_status=130
  fi
  trap - INT TERM
  printf '\nValidation interrupted by %s; stopping active workers.\n' \
    "${signal_name}" >&2
  marl_validation_cancel_active
  marl_validation_cleanup
  exit "${exit_status}"
}

marl_validation_init() {
  local max_jobs="${1:-}"
  local suite_name="${2:-validation}"

  if [[ ! "${max_jobs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: validation concurrency must be a positive integer." >&2
    return 2
  fi
  if (( MARL_VALIDATION_MAX_JOBS != 0 )); then
    echo "error: validation process pool is already initialized." >&2
    return 2
  fi

  MARL_VALIDATION_MAX_JOBS="${max_jobs}"
  MARL_VALIDATION_LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/marl-${suite_name}.XXXXXX")"
  trap 'marl_validation_handle_signal INT' INT
  trap 'marl_validation_handle_signal TERM' TERM
  MARL_VALIDATION_TRAPS_INSTALLED=1
}

marl_validation_active_count() {
  printf '%s\n' "${#MARL_VALIDATION_TASK_BY_PID[@]}"
}

marl_validation_reap_one() {
  local -a active_pids=("${!MARL_VALIDATION_TASK_BY_PID[@]}")
  local completed_pid=""
  local wait_status=0
  local task_id=""
  local label=""
  local log_path=""
  local result_path=""
  local command_status=""
  local finished_at=""
  local started_at=""

  if (( ${#active_pids[@]} == 0 )); then
    return 0
  fi

  if wait -n -p completed_pid "${active_pids[@]}"; then
    wait_status=0
  else
    wait_status=$?
  fi
  if [[ -z "${completed_pid:-}" ]] || \
     [[ -z "${MARL_VALIDATION_TASK_BY_PID[${completed_pid:-}]+present}" ]]; then
    completed_pid="${active_pids[0]}"
    if wait "${completed_pid}"; then
      wait_status=0
    else
      wait_status=$?
    fi
  fi

  task_id="${MARL_VALIDATION_TASK_BY_PID[${completed_pid}]}"
  unset 'MARL_VALIDATION_TASK_BY_PID['"${completed_pid}"']'
  label="${MARL_VALIDATION_LABEL_BY_TASK[${task_id}]}"
  log_path="${MARL_VALIDATION_LOG_BY_TASK[${task_id}]}"
  result_path="${MARL_VALIDATION_RESULT_BY_TASK[${task_id}]}"

  if [[ -s "${result_path}" ]]; then
    IFS=' ' read -r command_status started_at finished_at < "${result_path}"
  else
    command_status="${wait_status}"
    started_at="$(date +%s)"
    finished_at="${started_at}"
  fi
  if [[ ! "${command_status}" =~ ^[0-9]+$ ]]; then
    command_status=1
  fi

  MARL_VALIDATION_STATUS_BY_TASK["${task_id}"]="${command_status}"
  MARL_VALIDATION_DURATION_BY_TASK["${task_id}"]=$((finished_at - started_at))
  printf '\n===== %s =====\n' "${label}"
  if [[ -s "${log_path}" ]]; then
    cat -- "${log_path}"
    printf '\n'
  fi
  printf '%s: exit %s, %ss\n' \
    "${label}" \
    "${command_status}" \
    "${MARL_VALIDATION_DURATION_BY_TASK[${task_id}]}"

  if (( command_status != 0 )); then
    MARL_VALIDATION_FAILURES=$((MARL_VALIDATION_FAILURES + 1))
  fi
  return 0
}

marl_validation_start() {
  local label="${1:-}"
  shift || true

  if (( MARL_VALIDATION_MAX_JOBS == 0 )); then
    echo "error: initialize the validation process pool before adding work." >&2
    return 2
  fi
  if [[ -z "${label}" || $# -eq 0 ]]; then
    echo "error: validation work requires a label and command." >&2
    return 2
  fi

  while (( $(marl_validation_active_count) >= MARL_VALIDATION_MAX_JOBS )); do
    marl_validation_reap_one
  done

  MARL_VALIDATION_NEXT_TASK=$((MARL_VALIDATION_NEXT_TASK + 1))
  local task_id="${MARL_VALIDATION_NEXT_TASK}"
  local log_path="${MARL_VALIDATION_LOG_DIR}/task-${task_id}.log"
  local result_path="${MARL_VALIDATION_LOG_DIR}/task-${task_id}.result"
  local started_at
  started_at="$(date +%s)"

  MARL_VALIDATION_TASK_ORDER+=("${task_id}")
  MARL_VALIDATION_LABEL_BY_TASK["${task_id}"]="${label}"
  MARL_VALIDATION_LOG_BY_TASK["${task_id}"]="${log_path}"
  MARL_VALIDATION_RESULT_BY_TASK["${task_id}"]="${result_path}"
  printf 'Starting %s\n' "${label}"

  (
    set +e
    (
      set -euo pipefail
      "$@"
    ) > "${log_path}" 2>&1
    command_status=$?
    finished_at="$(date +%s)"
    printf '%s %s %s\n' \
      "${command_status}" \
      "${started_at}" \
      "${finished_at}" > "${result_path}"
    exit "${command_status}"
  ) &
  local child_pid=$!
  MARL_VALIDATION_TASK_BY_PID["${child_pid}"]="${task_id}"
}

marl_validation_finish() {
  local task_id=""

  while (( $(marl_validation_active_count) > 0 )); do
    marl_validation_reap_one
  done

  printf '\n===== validation summary =====\n'
  for task_id in "${MARL_VALIDATION_TASK_ORDER[@]}"; do
    printf '%-48s exit %-3s %ss\n' \
      "${MARL_VALIDATION_LABEL_BY_TASK[${task_id}]}" \
      "${MARL_VALIDATION_STATUS_BY_TASK[${task_id}]:-1}" \
      "${MARL_VALIDATION_DURATION_BY_TASK[${task_id}]:-0}"
  done

  (( MARL_VALIDATION_FAILURES == 0 ))
}

marl_validation_cancel_active() {
  local pid=""

  for pid in "${!MARL_VALIDATION_TASK_BY_PID[@]}"; do
    marl_validation_terminate_process_tree "${pid}"
  done
  for pid in "${!MARL_VALIDATION_TASK_BY_PID[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  MARL_VALIDATION_TASK_BY_PID=()
}

marl_validation_cleanup() {
  if [[ -n "${MARL_VALIDATION_LOG_DIR}" && -d "${MARL_VALIDATION_LOG_DIR}" ]]; then
    rm -rf -- "${MARL_VALIDATION_LOG_DIR}"
  fi
  MARL_VALIDATION_TASK_BY_PID=()
  MARL_VALIDATION_LABEL_BY_TASK=()
  MARL_VALIDATION_LOG_BY_TASK=()
  MARL_VALIDATION_RESULT_BY_TASK=()
  MARL_VALIDATION_STATUS_BY_TASK=()
  MARL_VALIDATION_DURATION_BY_TASK=()
  MARL_VALIDATION_TASK_ORDER=()
  MARL_VALIDATION_MAX_JOBS=0
  MARL_VALIDATION_NEXT_TASK=0
  MARL_VALIDATION_FAILURES=0
  MARL_VALIDATION_LOG_DIR=""
  if (( MARL_VALIDATION_TRAPS_INSTALLED != 0 )); then
    trap - INT TERM
  fi
  MARL_VALIDATION_TRAPS_INSTALLED=0
}
