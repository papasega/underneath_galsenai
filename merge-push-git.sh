#!/usr/bin/env bash
# Publish the current branch, merge it into the target, then publish the target.
# Usage: ./merge-push-git.sh [target-branch=main] [remote=origin]
set -euo pipefail

if [[ $# -gt 2 || ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: $0 [target-branch=main] [remote=origin]"
  exit 0
fi
target=${1:-main}
remote=${2:-origin}
[[ $remote != -* ]] || { echo "Invalid remote: $remote" >&2; exit 1; }
git check-ref-format --branch "$target" >/dev/null
cd "$(dirname "$0")"
cd "$(git rev-parse --show-toplevel)"
source_branch=$(git symbolic-ref --quiet --short HEAD) || {
  echo "Switch to a branch before merging." >&2
  exit 1
}
if [[ $source_branch == "$target" ]]; then
  echo "Already on $target. Switch to the branch you want to merge first." >&2
  exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "Commit or stash all changes before merging." >&2
  exit 1
fi
git remote get-url "$remote" >/dev/null
git fetch "$remote"
if ! git show-ref --verify --quiet "refs/remotes/$remote/$target"; then
  echo "Target branch $remote/$target does not exist." >&2
  exit 1
fi
# Stop on diverging histories instead of rewriting commits.
if git show-ref --verify --quiet "refs/remotes/$remote/$source_branch"; then
  git merge --ff-only "$remote/$source_branch"
fi
git push -u "$remote" "$source_branch"
if git show-ref --verify --quiet "refs/heads/$target"; then
  git switch "$target"
else
  git switch --track -c "$target" "$remote/$target"
fi
git merge --ff-only "$remote/$target"
git merge --no-ff "$source_branch" -m "Merge branch '$source_branch' into $target"
git diff --check
git push "$remote" "$target"
echo "Published $source_branch and merged it into $remote/$target. Current branch: $target."
