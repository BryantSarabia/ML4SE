"""
Direct Python script to apply commit date changes using git filter-branch
"""
import subprocess
import sys

print("=" * 80)
print("Applying Commit Date Changes")
print("=" * 80)

# Read the mapping
commit_map = {}
with open('commit_date_map.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            commit_map[parts[0]] = parts[1]

print(f"Loaded {len(commit_map)} commit date mappings")

# Create the filter script
filter_script = "case \"$GIT_COMMIT\" in\n"
for commit_hash, date in commit_map.items():
    filter_script += f"  {commit_hash})\n"
    filter_script += f"    export GIT_AUTHOR_DATE='{date}'\n"
    filter_script += f"    export GIT_COMMITTER_DATE='{date}'\n"
    filter_script += f"    ;;\n"
filter_script += "esac\n"

# Write filter to temp file
with open('.git_filter_temp.sh', 'w') as f:
    f.write(filter_script)

print("\nRunning git filter-branch...")
print("This will rewrite commit history - please wait...\n")

try:
    # Run git filter-branch
    cmd = [
        'git', 'filter-branch', '-f',
        '--env-filter', filter_script,
        '--', '--all'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    print("Success!")
    print(result.stdout)
    
    if result.stderr:
        print("Messages:")
        print(result.stderr)
    
    print("\n" + "=" * 80)
    print("Commit dates have been updated!")
    print("=" * 80)
    print("\nVerify with:")
    print("  git log --format='%h %ai %s' | head -20")
    
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    print(f"stdout: {e.stdout}")
    print(f"stderr: {e.stderr}")
    sys.exit(1)
