#!/usr/bin/env python3
"""
Meta-runner script that keeps running the results script until completion.
It monitors the progress file to know when the job is truly done.

Usage:
    python run_until_complete.py
    
Or make it executable:
    chmod +x run_until_complete.py
    ./run_until_complete.py
"""

import subprocess
import os
import time
import sys
import json
from datetime import datetime

# Configuration
RESULTS_SCRIPT = "results.py"  # Change this to your script name
PROGRESS_FILE = "data/prestress_progress.json"
MAX_ATTEMPTS = 100  # Maximum number of restart attempts
WAIT_TIME = 5  # Seconds to wait before checking if script is still running
CHECK_INTERVAL = 2  # How often to check if subprocess is alive
LOG_FILE = "run_until_complete.log"

def log_message(message):
    """Log message with timestamp to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + '\n')

def get_progress():
    """Get current progress from progress file"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def run_script_with_monitoring():
    """Run the script and monitor its progress"""
    log_message(f"Starting {RESULTS_SCRIPT}")
    
    # Start the subprocess
    process = subprocess.Popen(
        [sys.executable, RESULTS_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor the process
    while True:
        # Check if process is still running
        poll_result = process.poll()
        
        if poll_result is not None:
            # Process has finished
            return poll_result
        
        # Read and print any output
        line = process.stdout.readline()
        if line:
            print(line.rstrip())
        
        time.sleep(0.1)

def main():
    """Main runner loop"""
    log_message("="*60)
    log_message("Meta-runner started")
    log_message(f"Will run {RESULTS_SCRIPT} until completion")
    log_message(f"Max attempts: {MAX_ATTEMPTS}")
    log_message("="*60)
    
    attempt = 0
    last_progress = None
    
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        
        # Check if progress file exists (indicates work still to do)
        if not os.path.exists(PROGRESS_FILE):
            # Check if this is because we're done or haven't started
            if attempt > 1:
                log_message("Progress file removed - work completed successfully!")
                log_message(f"Total attempts needed: {attempt - 1}")
                break
            else:
                log_message("No progress file found - starting fresh")
        else:
            progress = get_progress()
            if progress:
                log_message(f"Resuming from: Batch {progress['batch']}, Run {progress['run']}")
        
        log_message(f"\nAttempt {attempt}/{MAX_ATTEMPTS}")
        log_message("-" * 40)
        
        # Run the script
        try:
            exit_code = run_script_with_monitoring()
            
            if exit_code == 0:
                log_message(f"Script exited normally (code {exit_code})")
                
                # Give it a moment to clean up files
                time.sleep(2)
                
                # Check if progress file is gone (indicating completion)
                if not os.path.exists(PROGRESS_FILE):
                    log_message("SUCCESS: All work completed!")
                    break
                else:
                    log_message("Progress file still exists - may have more work to do")
            else:
                log_message(f"Script exited with error code {exit_code}")
                
        except KeyboardInterrupt:
            log_message("\n! Interrupted by user")
            sys.exit(1)
            
        except Exception as e:
            log_message(f"! Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Check progress to see if we're making headway
        current_progress = get_progress()
        if current_progress and last_progress:
            if (current_progress['batch'] == last_progress['batch'] and 
                current_progress['run'] == last_progress['run']):
                log_message("Warning: No progress made in last run")
        last_progress = current_progress
        
        # Wait before next attempt
        if os.path.exists(PROGRESS_FILE):
            log_message(f"Waiting {WAIT_TIME} seconds before next attempt...")
            time.sleep(WAIT_TIME)
    
    # Final check
    if os.path.exists(PROGRESS_FILE):
        log_message(f"\nFAILED: Reached maximum attempts ({MAX_ATTEMPTS})")
        log_message("Progress file still exists - work incomplete")
        progress = get_progress()
        if progress:
            log_message(f"Stopped at: Batch {progress['batch']}, Run {progress['run']}")
        sys.exit(1)
    else:
        log_message("\nAll done! Results script completed successfully.")
        log_message(f"Check {LOG_FILE} for full execution history")
        sys.exit(0)

if __name__ == "__main__":
    main()