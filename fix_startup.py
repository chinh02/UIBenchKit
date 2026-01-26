# Add module-level initialization to load runs on startup

with open('/home/ec2-user/dcgen/api.py', 'r') as f:
    content = f.read()

# Check if already patched
if '# Load runs on module import (for gunicorn)' in content:
    print('Already patched')
else:
    # Find the end of load_existing_runs function and add initialization after it
    old_pattern = '''    return loaded


def run_evaluation_for_run'''
    
    new_pattern = '''    return loaded


# Load runs on module import (for gunicorn)
# This ensures runs persist across restarts
_loaded_count = load_existing_runs()
print(f"[DCGen] Loaded {_loaded_count} existing runs from disk")


def run_evaluation_for_run'''

    content = content.replace(old_pattern, new_pattern)
    
    with open('/home/ec2-user/dcgen/api.py', 'w') as f:
        f.write(content)
    
    print('Patched api.py to load runs on startup')
