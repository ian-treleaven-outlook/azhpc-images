#!/bin/bash
############################################################################
# logger.sh
#
# Shared logging helpers for azhpc-images bash scripts.
#
# Source this file from a script:
#     source "${UTILS_DIR}/logger.sh"
#
# Configuration (override via environment variables before sourcing):
#   LOG_FORMAT     text | json   (default: text if stdout is a TTY, else json)
#   LOG_LEVEL      info | debug  (default: info)
#   LOG_FILE       path          (default: empty; when set, also appends here)
#   RUN_ID         string        (default: ADO BUILD_BUILDID, else a UUID)
############################################################################

: "${LOG_FORMAT:=}"
: "${LOG_LEVEL:=info}"
: "${LOG_FILE:=}"
: "${RUN_ID:=${BUILD_BUILDID:-$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo unknown)}}"
: "${IMPLEMENTATION:=bash}"
: "${VERSION:=}"
: "${BUILD_COMMIT:=}"
: "${VENDOR:=}"
: "${GPU:=}"
: "${OS:=}"
: "${FIPS:=false}"

if [[ -z "${LOG_FORMAT}" ]]; then
    if [[ -t 1 ]]; then LOG_FORMAT=text; else LOG_FORMAT=json; fi
fi

# Redact known token-shaped values
_redact(){
    local s=$1
    # Bearer tokens in URLs / headers
    s=$(echo "$s" | sed -E 's#(bearer[[:space:]]+)[A-Za-z0-9._\-]+#\1[REDACTED]#gi')
    # Common token-shaped query params
    s=$(echo "$s" | sed -E 's#((sig|sas|token|key)=)[A-Za-z0-9._%\-]+#\1[REDACTED]#gi')
    printf '%s' "$s"
}

# Internal: emit a single log record. Callers should use log_info/warn/debug/error.
_log(){
    local level=$1
    local op=$2
    local message
    message=$(_redact "$3")

    if [[ "${level}" = "debug" && "${LOG_LEVEL}" != "debug" ]]; then
        return 0
    fi

    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    local line
    if [[ "${LOG_FORMAT}" = "json" ]]; then
        line=$(jq -cn \
            --arg ts "${ts}" \
            --arg level "${level}" \
            --arg op "${op}" \
            --arg msg "${message}" \
            --arg run_id "${RUN_ID}" \
            --arg impl "${IMPLEMENTATION}" \
            --arg version "${VERSION}" \
            --arg commit "${BUILD_COMMIT}" \
            --arg vendor "${VENDOR}" \
            --arg gpu "${GPU}" \
            --arg os_name "${OS}" \
            --argjson fips "${FIPS}" \
            '{ts:$ts, level:$level, op:$op, msg:$msg,
            run_id:$run_id, implementation:$impl,
            version:$version, commit:$commit,
            vendor:$vendor, gpu:$gpu, os:$os_name, fips:$fips}')
    else
        local upper_level
        upper_level=$(echo "${level}" | tr '[:lower:]' '[:upper:]')
        printf -v line '%s  %-5s  %-14s %s' "${ts}" "${upper_level}" "${op}" "${message}"
    fi

    printf '%s\n' "${line}"
    if [[ -n "${LOG_FILE}" ]]; then
        printf '%s\n' "${line}" >> "${LOG_FILE}"
    fi
}

############################################################################
# @Brief    : Log an INFO-level message describing a high-level operation.
# @Args     : (1) op       Operation name in kebab-case (e.g. install-cmake)
#             (2) message  Single-line description of what is happening
# @RetVal   : 0
############################################################################
log_info(){
    _log info "$1" "$2"
}

############################################################################
# @Brief    : Log a WARN-level message; unexpected but the build continues.
# @Args     : (1) op       (2) message
# @RetVal   : 0
############################################################################
log_warn(){
    _log warn "$1" "$2"
}

############################################################################
# @Brief    : Log a DEBUG-level message; only emitted when LOG_LEVEL=debug.
# @Args     : (1) op       (2) message
# @RetVal   : 0
############################################################################
log_debug(){
    _log debug "$1" "$2"
}

############################################################################
# @Brief    : Log an ERROR-level message; caller decides whether to exit.
# @Args     : (1) op       (2) message
# @RetVal   : 0
############################################################################
log_error(){
    _log error "$1" "$2"
}

log_error_detail(){
    # $1 op, $2 message, $3 multi-line stderr or stack
    if [[ "${LOG_FORMAT}" = "json" ]]; then
        # emit a richer record with the detail field
        local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        jq -cn --arg ts "$ts" --arg op "$1" --arg msg "$(_redact "$2")" \
               --arg detail "$(_redact "$3")" \
            '{ts:$ts, level:"error", op:$op, msg:$msg, error_detail:$detail}'
    else
        log_error "$1" "$2"
        # indent the detail so each line is visually distinct
        printf '%s\n' "$3" | sed 's/^/    | /'
    fi
}