#!/bin/bash
# Script to redate commits based on commit_date_map.txt

# Read the map file into an associative array
declare -A DATE_MAP
while IFS=' ' read -r hash datetime; do
    DATE_MAP["$hash"]="$datetime"
done < commit_date_map.txt

# Backup current branch
git branch backup-before-redate

echo "Starting git filter-branch to redate commits..."

git filter-branch -f --env-filter '
COMMIT_HASH=$GIT_COMMIT
NEW_DATE="${DATE_MAP[$COMMIT_HASH]}"

if [ -n "$NEW_DATE" ]; then
    export GIT_AUTHOR_DATE="$NEW_DATE"
    export GIT_COMMITTER_DATE="$NEW_DATE"
fi
' -- --all

echo "Done! Commit dates have been updated."
echo "Backup branch created: backup-before-redate"
echo ""
echo "To verify: git log --format='%h %ai %s' | head -20"
echo "To restore if needed: git reset --hard backup-before-redate"
