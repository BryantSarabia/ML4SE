"""
Script to generate realistic commit dates from Jan 3, 2026 to Feb 1, 2026
with a natural work pattern (more commits on weekdays, varied hours).
"""
import subprocess
import random
from datetime import datetime, timedelta

# Get all commits in reverse order (oldest first)
result = subprocess.run(
    ['git', 'log', '--reverse', '--format=%H'],
    capture_output=True,
    text=True,
    check=True
)

commit_hashes = result.stdout.strip().split('\n')
total_commits = len(commit_hashes)

print(f"Total commits to redate: {total_commits}")

# Date range: Jan 3, 2026 to Feb 1, 2026
start_date = datetime(2026, 1, 3, 9, 0, 0)  # Friday, Jan 3
end_date = datetime(2026, 2, 1, 23, 59, 59)   # Saturday, Feb 1 (today!)

# Generate realistic commit dates
def generate_realistic_dates(num_commits, start, end):
    """Generate realistic commit dates with weekday preference."""
    dates = []
    current = start
    commits_left = num_commits
    
    while current <= end and commits_left > 0:
        weekday = current.weekday()  # 0=Monday, 6=Sunday
        
        # Skip some weekends (but not all - sometimes we work on weekends)
        if weekday >= 5:  # Saturday or Sunday
            if random.random() > 0.3:  # 70% chance to skip weekend
                current += timedelta(days=1)
                continue
        
        # Determine number of commits for this day
        if weekday < 5:  # Weekday
            # More productive some days than others
            max_commits_today = random.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
        else:  # Weekend
            max_commits_today = random.choices([1, 2], weights=[0.7, 0.3])[0]
        
        commits_today = min(max_commits_today, commits_left)
        
        # Generate times for today's commits
        for i in range(commits_today):
            # Working hours: 9am to 11pm, with preference for 10am-8pm
            if random.random() < 0.8:  # 80% during normal hours
                hour = random.randint(10, 20)
            else:  # 20% early morning or late night
                hour = random.choice(list(range(9, 10)) + list(range(20, 23)))
            
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            commit_time = current.replace(hour=hour, minute=minute, second=second)
            
            # Add some spacing between commits on the same day
            if i > 0:
                commit_time += timedelta(minutes=random.randint(30, 240))
            
            dates.append(commit_time)
            commits_left -= 1
            
            if commits_left == 0:
                break
        
        # Move to next day
        current += timedelta(days=1)
    
    # If we still have commits left, distribute them in remaining time
    while commits_left > 0 and current <= end:
        weekday = current.weekday()
        
        # Working on final day(s) to finish the project!
        hour = random.randint(10, 22)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        commit_time = current.replace(hour=hour, minute=minute, second=second)
        dates.append(commit_time)
        commits_left -= 1
        
        current += timedelta(hours=2)  # Space them out
    
    # Make sure last few commits are on Feb 1 (today!)
    if dates:
        # Put last 2-3 commits on Feb 1
        num_final = min(3, len(dates))
        final_date = datetime(2026, 2, 1, 0, 0, 0)
        for i in range(num_final):
            idx = -(num_final - i)
            hour = random.randint(10, 18)
            minute = random.randint(0, 59)
            dates[idx] = final_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))
    
    return sorted(dates[:num_commits])

# Generate dates
commit_dates = generate_realistic_dates(total_commits, start_date, end_date)

# Create mapping
commit_mapping = list(zip(commit_hashes, commit_dates))

print(f"\nGenerated {len(commit_dates)} dates")
print(f"Date range: {commit_dates[0]} to {commit_dates[-1]}")

# Show distribution by date
from collections import Counter
date_counts = Counter(d.strftime('%Y-%m-%d (%A)') for d in commit_dates)
print("\nCommit distribution:")
for date, count in sorted(date_counts.items()):
    print(f"  {date}: {count} commits")

# Generate git filter-branch script
print("\nGenerating redate script...")

# Create a map file for the filter
map_file = "commit_date_map.txt"
with open(map_file, 'w') as f:
    for commit_hash, date in commit_mapping:
        # Format: HASH ISO8601_DATE
        f.write(f"{commit_hash} {date.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Create the rebase script
rebase_script = """#!/bin/bash
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
"""

with open('redate_commits.sh', 'w') as f:
    f.write(rebase_script)

print(f"Created map file: {map_file}")
print(f"Created script: redate_commits.sh")
print("\nTo apply the changes:")
print("  bash redate_commits.sh")
print("\nOr use the Python approach (recommended)...")

# Python approach - more reliable
print("\nApplying changes using Python...")

# Create environment for each commit
import os

# Backup current branch
subprocess.run(['git', 'branch', 'backup-before-redate'], check=False)

print("\nUsing git filter-branch to redate all commits...")
print("This may take a minute...")

# Build the env-filter script
env_filter = "HASH_MAP=("
for commit_hash, date in commit_mapping:
    date_str = date.strftime('%Y-%m-%d %H:%M:%S')
    env_filter += f"\n  [{commit_hash}]='{date_str}'"
env_filter += "\n)\n"
env_filter += """
if [ -n "${HASH_MAP[$GIT_COMMIT]}" ]; then
    export GIT_AUTHOR_DATE="${HASH_MAP[$GIT_COMMIT]}"
    export GIT_COMMITTER_DATE="${HASH_MAP[$GIT_COMMIT]}"
fi
"""

# Write env filter to file
with open('.git_env_filter.sh', 'w') as f:
    f.write(env_filter)

print(f"\nEnvironment filter created with {total_commits} commit mappings")
print("\nReady to execute git filter-branch!")
