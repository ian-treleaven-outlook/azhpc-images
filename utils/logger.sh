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

# When no LOG_FILE was provided, give this run its own timestamped log file so
# every invocation is captured separately. LOG_DIR is overridable via the
# environment. Exporting LOG_FILE lets child component scripts (which re-source
# this file) inherit the same path, so a whole run lands in one file.
if [[ -z "${LOG_FILE}" ]]; then
    : "${LOG_DIR:=/var/log/azhpc}"
    if mkdir -p "${LOG_DIR}" 2>/dev/null; then
        LOG_FILE="${LOG_DIR}/azhpc_$(date +%Y%m%d-%H%M%S).log"
    fi
fi
export LOG_FILE

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

        # Short HH:MM:SS for terminal; full ISO ts kept for the file
        local ts_short=${ts:11:8}

        # Colorize the level only when stdout is a TTY
        local lvl_pretty="${upper_level}"
        if [[ -t 1 ]]; then
            local c_reset=$'\e[0m'
            case "${level}" in
                debug) lvl_pretty=$'\e[2;37m'"${upper_level}${c_reset}" ;;  # dim grey
                info)  lvl_pretty=$'\e[36m'"${upper_level}${c_reset}"   ;;  # cyan
                warn)  lvl_pretty=$'\e[33m'"${upper_level}${c_reset}"   ;;  # yellow
                error) lvl_pretty=$'\e[31m'"${upper_level}${c_reset}"   ;;  # red
            esac
        fi

        printf -v line     '%s  %-14b  %-14s | %s' "${ts_short}" "${lvl_pretty}" "${op}" "${message}"
        printf -v line_raw '%s  %-5s  %-14s | %s' "${ts}"        "${upper_level}" "${op}" "${message}"
    fi

    printf '%s\n' "${line}"
    if [[ -n "${LOG_FILE}" ]]; then
        # File gets the uncolored, full-timestamp version
        printf '%s\n' "${line_raw:-$line}" >> "${LOG_FILE}"
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

############################################################################
# @Brief    : Log an ERROR-level message together with a multi-line detail
#             block (e.g. captured stderr or a stack trace). In JSON mode the
#             detail is emitted as an `error_detail` field on the record; in
#             text mode the detail is printed below the error, indented.
# @Args     : (1) op       Operation name in kebab-case
#             (2) message  Single-line summary of the error
#             (3) detail   Multi-line stderr or stack trace
# @RetVal   : 0
############################################################################
log_error_detail(){
    if [[ "${LOG_FORMAT}" = "json" ]]; then
        local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        jq -cn --arg ts "$ts" --arg op "$1" --arg msg "$(_redact "$2")" \
               --arg detail "$(_redact "$3")" \
            '{ts:$ts, level:"error", op:$op, msg:$msg, error_detail:$detail}'
    else
        log_error "$1" "$2"
        printf '%s\n' "$3" | sed 's/^/    | /'
    fi
}