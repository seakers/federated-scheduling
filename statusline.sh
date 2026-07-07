#!/bin/bash
input=$(cat)

# Check if jq is available
HAS_JQ=false
if command -v jq >/dev/null 2>&1; then
    HAS_JQ=true
fi

# Prefer Windows curl for SSL certificate handling
CURL="curl"
if [ -x "/c/Windows/System32/curl.exe" ]; then
    CURL="/c/Windows/System32/curl.exe"
fi

MODEL=$(echo "$input" | jq -r '.model.display_name')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')
# NOTE: Cost display is commented out by default because it uses estimates from Claude Code's
# internal implementation which may not reflect actual govcloud pricing. Uncomment if you want
# to see estimated costs, but be aware they may not be accurate for your environment.
# COST=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
DURATION_MS=$(echo "$input" | jq -r '.cost.total_duration_ms // 0')

CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'

# Configurable budget warning threshold (default 90%)
BUDGET_THRESHOLD=${JPL_GENAI_STATUSLINE_BUDGET_WARNING_THRESHOLD:-90}

# Determine environment from ANTHROPIC_BASE_URL
ENV_LABEL=""
if [ -n "$ANTHROPIC_BASE_URL" ]; then
  if [[ "$ANTHROPIC_BASE_URL" == *"gov2.genai-api"* ]]; then
    ENV_LABEL="GovDev | "
  elif [[ "$ANTHROPIC_BASE_URL" == *"gov.genai-api"* ]]; then
    ENV_LABEL="GovCloud | "
  elif [[ "$ANTHROPIC_BASE_URL" == *"com2.genai-api"* ]]; then
    ENV_LABEL="ComDev | "
  elif [[ "$ANTHROPIC_BASE_URL" == *"com.genai-api"* ]]; then
    ENV_LABEL="Commercial Cloud | "
  fi
fi

# Pick bar color based on context usage
if [ "$PCT" -ge 90 ]; then BAR_COLOR="$RED"
elif [ "$PCT" -ge 70 ]; then BAR_COLOR="$YELLOW"
else BAR_COLOR="$GREEN"; fi

# Determine whether to use ASCII or Unicode characters
# Force ASCII if:
# 1. Environment variable is set
# 2. Terminal doesn't support UTF-8
USE_ASCII=false

if [ -n "${JPL_GENAI_STATUSLINE_ASCII:-}" ]; then
  USE_ASCII=true
elif [[ ! "$LANG" =~ UTF-8 ]] && [[ ! "$LC_ALL" =~ UTF-8 ]]; then
  USE_ASCII=true
fi

if [ "$USE_ASCII" = true ]; then
  FILLED_CHAR='#'
  EMPTY_CHAR='-'
else
  FILLED_CHAR='█'
  EMPTY_CHAR='░'
fi

FILLED=$((PCT / 10)); EMPTY=$((10 - FILLED))
BAR_FILLED=$(printf "%${FILLED}s")
BAR_FILLED=${BAR_FILLED// /$FILLED_CHAR}
BAR_EMPTY=$(printf "%${EMPTY}s")
BAR_EMPTY=${BAR_EMPTY// /$EMPTY_CHAR}
BAR="${BAR_FILLED}${BAR_EMPTY}"

MINS=$((DURATION_MS / 60000)); SECS=$(((DURATION_MS % 60000) / 1000))

BRANCH=""
git rev-parse --git-dir > /dev/null 2>&1 && BRANCH=" | $(git branch --show-current 2>/dev/null)"

# Fetch budget info from subscription endpoint
BUDGET_BAR=""
BUDGET_INFO=""
BUDGET_WARNING=""

# Get API key from apiKeyHelper configured in settings.json
API_KEY_HELPER=""
if [ -f "$HOME/.claude/settings.json" ]; then
  if [ "$HAS_JQ" = true ]; then
    API_KEY_HELPER=$(jq -r '.apiKeyHelper // empty' "$HOME/.claude/settings.json" 2>/dev/null)
  else
    # Fallback: pure bash regex parsing
    settings_json=$(cat "$HOME/.claude/settings.json" | tr -d '\n\r\t' | sed 's/ \+/ /g')
    if [[ $settings_json =~ \"apiKeyHelper\"[[:space:]]*:[[:space:]]*\"([^\"]+)\" ]]; then
      API_KEY_HELPER="${BASH_REMATCH[1]}"
    fi
  fi
fi
API_KEY=""
if [ -n "$API_KEY_HELPER" ]; then
  # Expand tilde in path
  API_KEY_HELPER="${API_KEY_HELPER/#\~/$HOME}"
  # Strip leading/trailing quotes
  API_KEY_HELPER="${API_KEY_HELPER%\"}"
  API_KEY_HELPER="${API_KEY_HELPER#\"}"
  API_KEY=$("$API_KEY_HELPER" 2>/dev/null)
fi

if [ -n "$API_KEY" ]; then
  RESPONSE=$($CURL -s -X GET "${ANTHROPIC_BASE_URL}/subscription" \
    -H "Authorization: Bearer ${API_KEY}" \
    --max-time 2 2>/dev/null)

  if [ $? -eq 0 ]; then
    SUBSCRIPTION_ID=$(echo "$RESPONSE" | jq -r '.subscription_id // empty')
    KEY_SPEND=$(echo "$RESPONSE" | jq -r '.spend // empty')
    MAX_BUDGET=$(echo "$RESPONSE" | jq -r '.max_budget // empty')

    if [ -n "$KEY_SPEND" ] && [ -n "$MAX_BUDGET" ] && [ "$MAX_BUDGET" != "0" ]; then
      BUDGET_PCT=$(awk "BEGIN {printf \"%.0f\", ($KEY_SPEND / $MAX_BUDGET) * 100}")

      # Pick bar color based on budget usage
      if [ "$BUDGET_PCT" -ge 90 ]; then BUDGET_BAR_COLOR="$RED"
      elif [ "$BUDGET_PCT" -ge 70 ]; then BUDGET_BAR_COLOR="$YELLOW"
      else BUDGET_BAR_COLOR="$GREEN"; fi

      BUDGET_FILLED=$((BUDGET_PCT / 10)); BUDGET_EMPTY=$((10 - BUDGET_FILLED))
      BUDGET_BAR_FILLED=$(printf "%${BUDGET_FILLED}s")
      BUDGET_BAR_FILLED=${BUDGET_BAR_FILLED// /$FILLED_CHAR}
      BUDGET_BAR_EMPTY=$(printf "%${BUDGET_EMPTY}s")
      BUDGET_BAR_EMPTY=${BUDGET_BAR_EMPTY// /$EMPTY_CHAR}
      BUDGET_BAR="${BUDGET_BAR_COLOR}${BUDGET_BAR_FILLED}${BUDGET_BAR_EMPTY}${RESET}"
      KEY_SPEND_FMT=$(printf '%.2f' "$KEY_SPEND")
      MAX_BUDGET_FMT=$(printf '%.2f' "$MAX_BUDGET")
      BUDGET_INFO=" ${BUDGET_PCT}% | \$${KEY_SPEND_FMT}/\$${MAX_BUDGET_FMT}"

      # Display warning with Jira link if threshold reached
      if [ "$BUDGET_PCT" -ge "$BUDGET_THRESHOLD" ] && [ -n "$SUBSCRIPTION_ID" ] && [ -n "${JPL_GENAI_STATUSLINE_JIRA_BUDGET_REQUEST_URL:-}" ]; then
        JIRA_URL="${JPL_GENAI_STATUSLINE_JIRA_BUDGET_REQUEST_URL}?description=${SUBSCRIPTION_ID}&summary=Request%20Higher%20Budget"
        BUDGET_WARNING="${RED}WARNING: Budget limit approaching!${RESET} Request increase: ${CYAN}${JIRA_URL}${RESET}"
      fi
    fi
  fi
fi

# Extract basename - handle both Windows (\) and Unix (/) paths
BASENAME="${DIR##*\\}"  # Strip up to last backslash
BASENAME="${BASENAME##*/}"  # Strip up to last forward slash if no backslash was found
printf '%b\n' "${ENV_LABEL}${CYAN}[$MODEL]${RESET} ${BASENAME}$BRANCH"
# Uncomment the lines below to display estimated cost (see note above about accuracy)
# COST_FMT=$(printf '$%.2f' "$COST")
# echo -e "Ctx: ${BAR_COLOR}${BAR}${RESET} ${PCT}% | Est: ${YELLOW}${COST_FMT}${RESET} | Time: ${MINS}m ${SECS}s"
printf '%b\n' "Ctx: ${BAR_COLOR}${BAR}${RESET} ${PCT}% | Time: ${MINS}m ${SECS}s"

if [ -n "$BUDGET_BAR" ]; then
  printf '%b\n' "Budget: ${BUDGET_BAR}${BUDGET_INFO}"
fi
if [ -n "$BUDGET_WARNING" ]; then
  printf '%b\n' "${BUDGET_WARNING}"
fi
