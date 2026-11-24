#!/usr/bin/env bash
# Commit all non-ignored changes in the repository containing this script.
set -euo pipefail

if [[ $# -ne 1 || -z ${1//[[:space:]]/} ]]; then
  echo "Usage: $0 \"Commit message\"" >&2
  exit 1
fi
cd "$(dirname "$0")"
cd "$(git rev-parse --show-toplevel)"
git symbolic-ref --quiet --short HEAD >/dev/null || {
  echo "Switch to a branch before committing." >&2
  exit 1
}
git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi
git diff --cached --check
git diff --cached --stat
git commit -m "$1"
git status --short
