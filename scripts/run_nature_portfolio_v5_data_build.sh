#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-gold}"

EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/mnt/d/aspr_nature_portfolio_v5}"
USE_SYMLINKS="${USE_SYMLINKS:-1}"
MIGRATE_EXISTING="${MIGRATE_EXISTING:-0}"
NATURE_MARKDOWN_LINK="${NATURE_MARKDOWN_LINK:-data/nature_markdown}"
NATURE_MARKDOWN_DIR="${NATURE_MARKDOWN_DIR:-/mnt/d/aspr_nature_markdown}"

OUT_DIR="${OUT_DIR:-outputs/nature_portfolio_v5}"
CORPUS_DIR="${CORPUS_DIR:-data/knowledge_corpus/v5_nature_portfolio_full}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"

START_YEAR="${START_YEAR:-1980}"
END_YEAR="${END_YEAR:-2025}"
TAU="${TAU:-8}"

case "$PROFILE" in
    gold)
        DEFAULT_MAX_SOURCES=""
        DEFAULT_MAX_WORKS_PER_SOURCE="25000"
        DEFAULT_MAX_REFS=""
        DEFAULT_MAX_EDGES=""
        DEFAULT_MAX_FUTURE_PAPERS=""
        DEFAULT_MAX_CITERS_PER_WORK="1000"
        DEFAULT_MIN_BROAD_CATEGORIES="10"
        DEFAULT_MIN_FINE_DOMAINS="100"
        DEFAULT_MIN_BROAD_ELIGIBLE="2000"
        DEFAULT_MIN_DOMAIN_ELIGIBLE="200"
        DEFAULT_MIN_PAPERS_PER_DOMAIN="200"
        ;;
    silver)
        DEFAULT_MAX_SOURCES=""
        DEFAULT_MAX_WORKS_PER_SOURCE="5000"
        DEFAULT_MAX_REFS=""
        DEFAULT_MAX_EDGES=""
        DEFAULT_MAX_FUTURE_PAPERS=""
        DEFAULT_MAX_CITERS_PER_WORK="500"
        DEFAULT_MIN_BROAD_CATEGORIES="10"
        DEFAULT_MIN_FINE_DOMAINS="80"
        DEFAULT_MIN_BROAD_ELIGIBLE="2000"
        DEFAULT_MIN_DOMAIN_ELIGIBLE="200"
        DEFAULT_MIN_PAPERS_PER_DOMAIN="200"
        ;;
    smoke)
        DEFAULT_MAX_SOURCES="3"
        DEFAULT_MAX_WORKS_PER_SOURCE="100"
        DEFAULT_MAX_REFS="500"
        DEFAULT_MAX_EDGES=""
        DEFAULT_MAX_FUTURE_PAPERS="100"
        DEFAULT_MAX_CITERS_PER_WORK="100"
        DEFAULT_MIN_BROAD_CATEGORIES="1"
        DEFAULT_MIN_FINE_DOMAINS="1"
        DEFAULT_MIN_BROAD_ELIGIBLE="1"
        DEFAULT_MIN_DOMAIN_ELIGIBLE="1"
        DEFAULT_MIN_PAPERS_PER_DOMAIN="1"
        ;;
    *)
        echo "未知 PROFILE=$PROFILE；请使用 gold、silver 或 smoke。" >&2
        exit 2
        ;;
esac

MAX_SOURCES="${MAX_SOURCES:-$DEFAULT_MAX_SOURCES}"
MAX_WORKS_PER_SOURCE="${MAX_WORKS_PER_SOURCE:-$DEFAULT_MAX_WORKS_PER_SOURCE}"
MAX_REFS="${MAX_REFS:-$DEFAULT_MAX_REFS}"
MAX_EDGES="${MAX_EDGES:-$DEFAULT_MAX_EDGES}"
MAX_FUTURE_PAPERS="${MAX_FUTURE_PAPERS:-$DEFAULT_MAX_FUTURE_PAPERS}"
MAX_CITERS_PER_WORK="${MAX_CITERS_PER_WORK:-$DEFAULT_MAX_CITERS_PER_WORK}"

PER_PAGE="${PER_PAGE:-200}"
SOURCE_PER_QUERY="${SOURCE_PER_QUERY:-5}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"
MAX_RETRIES="${MAX_RETRIES:-5}"
OPENALEX_WORKERS="${OPENALEX_WORKERS:-2}"
SNAPSHOT_WORKERS="${SNAPSHOT_WORKERS:-8}"

REFRESH="${REFRESH:-0}"
REFRESH_ROSTER="${REFRESH_ROSTER:-0}"
OFFLINE_ROSTER="${OFFLINE_ROSTER:-0}"
SKIP_REFERENCE_FETCH="${SKIP_REFERENCE_FETCH:-0}"
SKIP_FUTURE_CITERS="${SKIP_FUTURE_CITERS:-0}"
USE_OPENALEX_SNAPSHOT="${USE_OPENALEX_SNAPSHOT:-1}"
OPENALEX_SNAPSHOT_DIR="${OPENALEX_SNAPSHOT_DIR:-/mnt/d/FabCitationData/openalex-snapshot}"
FETCH_SNAPSHOT_MISSING_ONLINE="${FETCH_SNAPSHOT_MISSING_ONLINE:-0}"

MIN_BROAD_CATEGORIES="${MIN_BROAD_CATEGORIES:-$DEFAULT_MIN_BROAD_CATEGORIES}"
MIN_FINE_DOMAINS="${MIN_FINE_DOMAINS:-$DEFAULT_MIN_FINE_DOMAINS}"
MIN_BROAD_ELIGIBLE="${MIN_BROAD_ELIGIBLE:-$DEFAULT_MIN_BROAD_ELIGIBLE}"
MIN_DOMAIN_ELIGIBLE="${MIN_DOMAIN_ELIGIBLE:-$DEFAULT_MIN_DOMAIN_ELIGIBLE}"
MIN_PAPERS_PER_DOMAIN="${MIN_PAPERS_PER_DOMAIN:-$DEFAULT_MIN_PAPERS_PER_DOMAIN}"

SOURCE_ROSTER="$OUT_DIR/nature_source_roster.csv"
SUBJECT_TAXONOMY="$OUT_DIR/nature_subject_taxonomy.csv"
TARGET_WORKS="$OUT_DIR/nature_target_works.csv"
REFERENCE_WORKS="$OUT_DIR/nature_reference_works.csv"
REFERENCE_EDGES="$OUT_DIR/nature_reference_edges.csv"
FUTURE_CITERS="$OUT_DIR/nature_future_citers.csv"
FUTURE_GRAPH_DELTAS="$OUT_DIR/nature_future_graph_deltas.csv"
MASTER_LOG="$LOG_DIR/nature_v5_data_build_$(date +%Y%m%d_%H%M%S).log"
CONFIG_SNAPSHOT="$OUT_DIR/nature_v5_data_build_config.env"
DOWNSTREAM_PATHS="$OUT_DIR/downstream_paths.env"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    mkdir -p "$LOG_DIR"
    echo "[$(timestamp)] $*" | tee -a "$MASTER_LOG"
}

append_optional_arg() {
    local -n cmd_ref="$1"
    local flag="$2"
    local value="$3"
    if [[ -n "$value" ]]; then
        cmd_ref+=("$flag" "$value")
    fi
}

source_roster_has_ids() {
    [[ -s "$SOURCE_ROSTER" ]] || return 1
    "$PYTHON_BIN" - "$SOURCE_ROSTER" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
try:
    frame = pd.read_csv(path, low_memory=False)
except Exception:
    raise SystemExit(1)
if "source_id" not in frame.columns:
    raise SystemExit(1)
count = frame["source_id"].fillna("").astype(str).str.strip().ne("").sum()
raise SystemExit(0 if count > 0 else 1)
PY
}

run_stage() {
    local stage="$1"
    shift
    local stage_log="$LOG_DIR/${stage}.log"
    log "开始阶段：$stage"
    log "执行命令：$*"
    set +e
    "$@" 2>&1 | tee -a "$stage_log" | tee -a "$MASTER_LOG"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -ne 0 ]]; then
        log "阶段失败：$stage，退出码=$status；详见日志 $stage_log"
        exit "$status"
    fi
    log "阶段完成：$stage；日志=$stage_log"
}

is_empty_dir() {
    [[ -d "$1" ]] && [[ -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

prepare_symlink() {
    local label="$1"
    local link_path="$2"
    local target_path="$3"
    local link_abs
    local target_abs
    link_abs="$(realpath -m "$link_path")"
    target_abs="$(realpath -m "$target_path")"
    if [[ "$link_abs" == "$target_abs" ]]; then
        mkdir -p "$target_path"
        echo "[$(timestamp)] 存储路径：$label 直接使用 $link_path"
        return
    fi
    mkdir -p "$(dirname "$link_path")" "$target_path"
    if [[ -L "$link_path" ]]; then
        local current_target
        current_target="$(readlink "$link_path")"
        if [[ "$(realpath -m "$current_target")" != "$target_abs" ]]; then
            rm "$link_path"
            ln -s "$target_path" "$link_path"
            echo "[$(timestamp)] 存储路径：$label 已重新链接 $link_path -> $target_path"
        else
            echo "[$(timestamp)] 存储路径：$label 已存在链接 $link_path -> $target_path"
        fi
        return
    fi
    if [[ -e "$link_path" ]]; then
        if is_empty_dir "$link_path"; then
            rmdir "$link_path"
            ln -s "$target_path" "$link_path"
            echo "[$(timestamp)] 存储路径：$label 已将空目录替换为软链接 $link_path -> $target_path"
            return
        fi
        if [[ "$MIGRATE_EXISTING" == "1" && -d "$link_path" ]]; then
            shopt -s dotglob nullglob
            mv "$link_path"/* "$target_path"/
            shopt -u dotglob nullglob
            rmdir "$link_path"
            ln -s "$target_path" "$link_path"
            echo "[$(timestamp)] 存储路径：$label 已迁移已有目录并建立软链接 $link_path -> $target_path"
            return
        fi
        echo "[$(timestamp)] 错误：$label 路径已存在且不是空目录：$link_path" >&2
        echo "可设置 MIGRATE_EXISTING=1 将其内容迁移到 $target_path，或指定其他 OUT_DIR/CORPUS_DIR。" >&2
        exit 2
    fi
    ln -s "$target_path" "$link_path"
    echo "[$(timestamp)] 存储路径：$label 已建立软链接 $link_path -> $target_path"
}

prepare_storage() {
    if [[ "$USE_SYMLINKS" != "1" ]]; then
        mkdir -p "$OUT_DIR" "$CORPUS_DIR" "$NATURE_MARKDOWN_LINK"
        echo "[$(timestamp)] 存储路径：已关闭软链接，直接使用仓库内路径"
        return
    fi
    prepare_symlink "openalex_outputs" "$OUT_DIR" "$EXTERNAL_DATA_ROOT/openalex_outputs"
    prepare_symlink "canonical_corpus" "$CORPUS_DIR" "$EXTERNAL_DATA_ROOT/corpus"
    prepare_symlink "nature_markdown" "$NATURE_MARKDOWN_LINK" "$NATURE_MARKDOWN_DIR"
}

write_config_snapshot() {
    mkdir -p "$OUT_DIR" "$LOG_DIR"
    cat > "$CONFIG_SNAPSHOT" <<EOF
PYTHON_BIN=$PYTHON_BIN
PROFILE=$PROFILE
EXTERNAL_DATA_ROOT=$EXTERNAL_DATA_ROOT
USE_SYMLINKS=$USE_SYMLINKS
MIGRATE_EXISTING=$MIGRATE_EXISTING
NATURE_MARKDOWN_LINK=$NATURE_MARKDOWN_LINK
NATURE_MARKDOWN_DIR=$NATURE_MARKDOWN_DIR
OUT_DIR=$OUT_DIR
CORPUS_DIR=$CORPUS_DIR
LOG_DIR=$LOG_DIR
START_YEAR=$START_YEAR
END_YEAR=$END_YEAR
TAU=$TAU
MAX_SOURCES=$MAX_SOURCES
MAX_WORKS_PER_SOURCE=$MAX_WORKS_PER_SOURCE
MAX_REFS=$MAX_REFS
MAX_EDGES=$MAX_EDGES
MAX_FUTURE_PAPERS=$MAX_FUTURE_PAPERS
MAX_CITERS_PER_WORK=$MAX_CITERS_PER_WORK
PER_PAGE=$PER_PAGE
SOURCE_PER_QUERY=$SOURCE_PER_QUERY
SLEEP_SECONDS=$SLEEP_SECONDS
TIMEOUT_SECONDS=$TIMEOUT_SECONDS
MAX_RETRIES=$MAX_RETRIES
OPENALEX_WORKERS=$OPENALEX_WORKERS
SNAPSHOT_WORKERS=$SNAPSHOT_WORKERS
REFRESH=$REFRESH
REFRESH_ROSTER=$REFRESH_ROSTER
OFFLINE_ROSTER=$OFFLINE_ROSTER
SKIP_REFERENCE_FETCH=$SKIP_REFERENCE_FETCH
SKIP_FUTURE_CITERS=$SKIP_FUTURE_CITERS
USE_OPENALEX_SNAPSHOT=$USE_OPENALEX_SNAPSHOT
OPENALEX_SNAPSHOT_DIR=$OPENALEX_SNAPSHOT_DIR
FETCH_SNAPSHOT_MISSING_ONLINE=$FETCH_SNAPSHOT_MISSING_ONLINE
MIN_BROAD_CATEGORIES=$MIN_BROAD_CATEGORIES
MIN_FINE_DOMAINS=$MIN_FINE_DOMAINS
MIN_BROAD_ELIGIBLE=$MIN_BROAD_ELIGIBLE
MIN_DOMAIN_ELIGIBLE=$MIN_DOMAIN_ELIGIBLE
MIN_PAPERS_PER_DOMAIN=$MIN_PAPERS_PER_DOMAIN
OPENALEX_EMAIL=${OPENALEX_EMAIL:-}
OPENALEX_API_KEY_SET=$([[ -n "${OPENALEX_API_KEY:-}" ]] && echo 1 || echo 0)
OPENALEX_API_KEYS_SET=$([[ -n "${OPENALEX_API_KEYS:-}" ]] && echo 1 || echo 0)
EOF
}

write_downstream_paths() {
    mkdir -p "$OUT_DIR"
    cat > "$DOWNSTREAM_PATHS" <<EOF
ASPR_NATURE_V5_PROFILE=$PROFILE
ASPR_NATURE_V5_OUTPUT_DIR=$OUT_DIR
ASPR_NATURE_V5_OUTPUT_REALPATH=$(realpath -m "$OUT_DIR")
ASPR_NATURE_V5_CORPUS_DIR=$CORPUS_DIR
ASPR_NATURE_V5_CORPUS_REALPATH=$(realpath -m "$CORPUS_DIR")
ASPR_NATURE_V5_WORKS=$CORPUS_DIR/works.csv
ASPR_NATURE_V5_CITATIONS=$CORPUS_DIR/citations.csv
ASPR_NATURE_V5_REFERENCE_WORKS=$CORPUS_DIR/reference_works.csv
ASPR_NATURE_V5_FUTURE_CITERS=$CORPUS_DIR/future_citers.csv
ASPR_NATURE_V5_FUTURE_GRAPH_DELTAS=$CORPUS_DIR/future_graph_deltas.csv
ASPR_NATURE_V5_SOURCE_ROSTER=$CORPUS_DIR/nature_source_roster.csv
ASPR_NATURE_V5_SUBJECT_TAXONOMY=$CORPUS_DIR/nature_subject_taxonomy.csv
ASPR_NATURE_V5_MANIFEST=$CORPUS_DIR/v5_nature_portfolio_full_manifest.json
ASPR_NATURE_V5_QUALITY_REPORT=$CORPUS_DIR/data_quality_report.json
ASPR_NATURE_MARKDOWN_ROOT=$NATURE_MARKDOWN_LINK
ASPR_NATURE_MARKDOWN_REALPATH=$(realpath -m "$NATURE_MARKDOWN_LINK")
EOF
}

verify_downstream_ready() {
    local missing=0
    local required_files=(
        "$CORPUS_DIR/works.csv"
        "$CORPUS_DIR/citations.csv"
        "$CORPUS_DIR/reference_works.csv"
        "$CORPUS_DIR/future_citers.csv"
        "$CORPUS_DIR/future_graph_deltas.csv"
        "$CORPUS_DIR/nature_source_roster.csv"
        "$CORPUS_DIR/nature_subject_taxonomy.csv"
        "$CORPUS_DIR/data_quality_report.json"
        "$CORPUS_DIR/v5_nature_portfolio_full_manifest.json"
        "$CORPUS_DIR/methods_nature_full_corpus.md"
    )
    for path in "${required_files[@]}"; do
        if [[ ! -s "$path" ]]; then
            log "错误：缺少后续代码所需产物：$path"
            missing=1
        fi
    done
    if [[ "$missing" -ne 0 ]]; then
        exit 2
    fi
    write_downstream_paths
    log "后续代码路径清单：$DOWNSTREAM_PATHS"
}

ensure_empty_future_files() {
    mkdir -p "$OUT_DIR"
    if [[ ! -s "$FUTURE_CITERS" ]]; then
        echo "paper_id,citer_id,citer_year,citer_primary_field,citer_primary_subfield,citer_primary_topic,fetch_status" > "$FUTURE_CITERS"
    fi
    if [[ ! -s "$FUTURE_GRAPH_DELTAS" ]]; then
        echo "paper_id,year,tau,n_future_citers,future_community_reach,future_field_reach,future_subfield_reach,future_field_entropy,future_topic_entropy,future_field_simpson,future_topic_simpson,future_first_year,future_last_year" > "$FUTURE_GRAPH_DELTAS"
    fi
}

prepare_storage
write_config_snapshot

log "Nature Portfolio v5 数据构建开始"
log "项目根目录：$PROJECT_ROOT"
log "构建档位：$PROFILE"
log "外部数据根目录：$EXTERNAL_DATA_ROOT"
log "配置快照：$CONFIG_SNAPSHOT"
if [[ -z "${OPENALEX_EMAIL:-}" ]]; then
    log "提醒：OPENALEX_EMAIL 为空。OpenAlex 可以继续访问，但设置邮箱通常有助于限流表现。"
fi

if [[ ! -s "$SOURCE_ROSTER" || "$REFRESH_ROSTER" == "1" ]] || ! source_roster_has_ids; then
    if [[ -s "$SOURCE_ROSTER" && "$REFRESH_ROSTER" != "1" ]]; then
        log "检测到已有 source roster 但没有 OpenAlex source_id，将自动重建：$SOURCE_ROSTER"
    fi
    roster_cmd=(
        "$PYTHON_BIN" -u scripts/build_nature_portfolio_source_roster.py
        --out-dir "$OUT_DIR"
        --per-query "$SOURCE_PER_QUERY"
        --sleep-seconds "$SLEEP_SECONDS"
        --timeout-seconds "$TIMEOUT_SECONDS"
        --max-retries "$MAX_RETRIES"
    )
    if [[ "$OFFLINE_ROSTER" == "1" ]]; then
        roster_cmd+=(--offline)
    fi
    run_stage "01_source_roster" "${roster_cmd[@]}"
else
    log "跳过阶段：01_source_roster，因为 $SOURCE_ROSTER 已存在。若需重建请设置 REFRESH_ROSTER=1。"
fi

if [[ "$OFFLINE_ROSTER" == "1" ]]; then
    log "停止：OFFLINE_ROSTER=1 只生成 taxonomy/seed roster；正式拉取论文需要 OpenAlex source_id。"
    exit 0
fi

target_cmd=(
    "$PYTHON_BIN" -u scripts/fetch_openalex_nature_works.py
    --source-roster "$SOURCE_ROSTER"
    --out-dir "$OUT_DIR"
    --checkpoint-dir "$OUT_DIR/checkpoints/target_works"
    --max-works-per-source "$MAX_WORKS_PER_SOURCE"
    --start-year "$START_YEAR"
    --end-year "$END_YEAR"
    --per-page "$PER_PAGE"
    --sleep-seconds "$SLEEP_SECONDS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --max-retries "$MAX_RETRIES"
    --workers "$OPENALEX_WORKERS"
)
append_optional_arg target_cmd --max-sources "$MAX_SOURCES"
if [[ "$REFRESH" == "1" ]]; then
    target_cmd+=(--refresh)
fi
run_stage "02_target_works" "${target_cmd[@]}"

if [[ "$SKIP_REFERENCE_FETCH" == "1" ]]; then
    reference_cmd=(
        "$PYTHON_BIN" -u scripts/build_reference_closure_v5.py
        --target-works "$TARGET_WORKS"
        --out-dir "$OUT_DIR"
        --checkpoint-jsonl "$OUT_DIR/checkpoints/reference_works.jsonl"
        --sleep-seconds "$SLEEP_SECONDS"
        --timeout-seconds "$TIMEOUT_SECONDS"
        --max-retries "$MAX_RETRIES"
        --workers "$OPENALEX_WORKERS"
        --skip-fetch
    )
    append_optional_arg reference_cmd --max-refs "$MAX_REFS"
    append_optional_arg reference_cmd --max-edges "$MAX_EDGES"
    run_stage "03_reference_closure" "${reference_cmd[@]}"
elif [[ "$USE_OPENALEX_SNAPSHOT" == "1" && -d "$OPENALEX_SNAPSHOT_DIR/data/works" ]]; then
    snapshot_reference_cmd=(
        "$PYTHON_BIN" -u scripts/build_reference_closure_v5_from_snapshot.py
        --target-works "$TARGET_WORKS"
        --reference-edges "$REFERENCE_EDGES"
        --out-dir "$OUT_DIR"
        --snapshot-dir "$OPENALEX_SNAPSHOT_DIR"
        --checkpoint-jsonl "$OUT_DIR/checkpoints/reference_works.jsonl"
        --workers "$SNAPSHOT_WORKERS"
    )
    append_optional_arg snapshot_reference_cmd --max-reference-ids "$MAX_REFS"
    append_optional_arg snapshot_reference_cmd --max-edges "$MAX_EDGES"
    run_stage "03_reference_closure_snapshot" "${snapshot_reference_cmd[@]}"
    if [[ "$FETCH_SNAPSHOT_MISSING_ONLINE" == "1" ]]; then
        reference_topup_cmd=(
            "$PYTHON_BIN" -u scripts/fetch_openalex_reference_missing_v5.py
            --retry-queue "$OUT_DIR/nature_reference_closure_api_retry_queue.csv"
            --reference-works "$REFERENCE_WORKS"
            --out-dir "$OUT_DIR"
            --checkpoint-csv "$OUT_DIR/checkpoints/reference_missing_online_success.csv"
            --failure-log "$OUT_DIR/checkpoints/reference_missing_online_failures.csv"
            --final-missing "$OUT_DIR/nature_reference_closure_final_missing_ids.csv"
            --sleep-seconds "$SLEEP_SECONDS"
            --timeout-seconds "$TIMEOUT_SECONDS"
            --max-retries "$MAX_RETRIES"
            --workers "$OPENALEX_WORKERS"
        )
        append_optional_arg reference_topup_cmd --max-refs "$MAX_REFS"
        run_stage "03b_reference_closure_online_missing" "${reference_topup_cmd[@]}"
    else
        log "跳过线上补抓 reference 缺失项：FETCH_SNAPSHOT_MISSING_ONLINE=0；缺失清单见 $OUT_DIR/nature_reference_closure_api_retry_queue.csv"
    fi
else
    log "未启用本地 OpenAlex snapshot，或路径不存在：$OPENALEX_SNAPSHOT_DIR；将退回线上 reference closure。"
    reference_cmd=(
        "$PYTHON_BIN" -u scripts/build_reference_closure_v5.py
        --target-works "$TARGET_WORKS"
        --out-dir "$OUT_DIR"
        --checkpoint-jsonl "$OUT_DIR/checkpoints/reference_works.jsonl"
        --sleep-seconds "$SLEEP_SECONDS"
        --timeout-seconds "$TIMEOUT_SECONDS"
        --max-retries "$MAX_RETRIES"
        --workers "$OPENALEX_WORKERS"
    )
    append_optional_arg reference_cmd --max-refs "$MAX_REFS"
    append_optional_arg reference_cmd --max-edges "$MAX_EDGES"
    run_stage "03_reference_closure" "${reference_cmd[@]}"
fi

if [[ "$SKIP_FUTURE_CITERS" == "1" ]]; then
    ensure_empty_future_files
    log "跳过阶段：04_future_citers，因为 SKIP_FUTURE_CITERS=1；已写入空的 label-only future 文件。"
else
    future_cmd=(
        "$PYTHON_BIN" -u scripts/build_future_citer_graph_v5.py
        --target-works "$TARGET_WORKS"
        --out-dir "$OUT_DIR"
        --checkpoint-dir "$OUT_DIR/checkpoints/future_citers_tau${TAU}"
        --tau "$TAU"
        --max-citers-per-work "$MAX_CITERS_PER_WORK"
        --per-page "$PER_PAGE"
        --sleep-seconds "$SLEEP_SECONDS"
        --timeout-seconds "$TIMEOUT_SECONDS"
        --max-retries "$MAX_RETRIES"
        --workers "$OPENALEX_WORKERS"
    )
    append_optional_arg future_cmd --max-papers "$MAX_FUTURE_PAPERS"
    if [[ "$REFRESH" == "1" ]]; then
        future_cmd+=(--refresh)
    fi
    run_stage "04_future_citers" "${future_cmd[@]}"
fi

materialize_cmd=(
    "$PYTHON_BIN" -u scripts/materialize_nature_full_corpus_v5.py
    --target-works "$TARGET_WORKS"
    --reference-works "$REFERENCE_WORKS"
    --reference-edges "$REFERENCE_EDGES"
    --future-citers "$FUTURE_CITERS"
    --future-graph-deltas "$FUTURE_GRAPH_DELTAS"
    --source-roster "$SOURCE_ROSTER"
    --subject-taxonomy "$SUBJECT_TAXONOMY"
    --corpus-dir "$CORPUS_DIR"
    --tau "$TAU"
    --min-papers-per-domain "$MIN_PAPERS_PER_DOMAIN"
    --min-broad-categories "$MIN_BROAD_CATEGORIES"
    --min-fine-domains "$MIN_FINE_DOMAINS"
    --min-broad-eligible "$MIN_BROAD_ELIGIBLE"
    --min-domain-eligible "$MIN_DOMAIN_ELIGIBLE"
)
run_stage "05_materialize" "${materialize_cmd[@]}"
verify_downstream_ready

log "Nature Portfolio v5 数据构建完成"
log "目标论文表：$TARGET_WORKS"
log "参考文献元数据表：$REFERENCE_WORKS"
log "引用边表：$REFERENCE_EDGES"
log "未来图谱 label 表：$FUTURE_GRAPH_DELTAS"
log "规范化本地语料目录：$CORPUS_DIR"
log "数据质量报告：$CORPUS_DIR/data_quality_report.json"
log "语料 manifest：$CORPUS_DIR/v5_nature_portfolio_full_manifest.json"
