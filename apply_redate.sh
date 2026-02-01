#!/bin/bash
# Apply the commit date changes using git filter-branch

echo "Backing up current refs..."
git update-ref refs/original/backup-main refs/heads/main

echo "Reading commit date mappings..."

# Read the entire mapping into a bash script
FILTER_SCRIPT=$(cat << 'FILTER_EOF'
# Commit hash to date mapping
case "$GIT_COMMIT" in
FILTER_EOF
)

# Add each commit mapping
while IFS=' ' read -r hash datetime; do
    FILTER_SCRIPT+="
    $hash)
        export GIT_AUTHOR_DATE='$datetime'
        export GIT_COMMITTER_DATE='$datetime'
        ;;"
done < commit_date_map.txt

FILTER_SCRIPT+="
esac
"

echo "Applying git filter-branch..."
echo "$FILTER_SCRIPT" > .git_filter_env.sh

git filter-branch -f --env-filter "$(cat .git_filter_env.sh)" -- --all

echo ""
echo "Done! Commit dates have been updated."
echo ""
echo "Verify with: git log --format='%h %ai %s' | head -20"
echo ""
echo "To restore if needed: git reset --hard refs/original/backup-main"
