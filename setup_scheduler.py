"""
Interactive Setup Script for Lightning.ai Auto-Restart Scheduler

Run this script to configure the auto-restart scheduler with your Lightning.ai credentials.
"""

import subprocess
import sys
import re
import os


def run_command(cmd, description="Running command"):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_lightning_cli():
    """Check if Lightning CLI is installed"""
    print("Checking for Lightning CLI...")
    success, stdout, stderr = run_command("lightning --version")
    
    if not success:
        print("❌ Lightning CLI not found")
        print("\nInstalling Lightning SDK...")
        success, _, _ = run_command("pip install lightning-sdk")
        
        if not success:
            print("❌ Failed to install Lightning SDK")
            print("Please run manually: pip install lightning-sdk")
            return False
        
        print("✓ Lightning SDK installed")
    else:
        print(f"✓ Lightning CLI found: {stdout.strip()}")
    
    return True


def check_authentication():
    """Check if authenticated with Lightning.ai"""
    print("\nChecking Lightning.ai authentication...")
    success, _, _ = run_command("lightning studio ls")
    
    if not success:
        print("❌ Not authenticated with Lightning.ai")
        print("\nPlease authenticate by running:")
        print("  lightning login")
        print("\nThen run this setup script again.")
        return False
    
    print("✓ Authenticated with Lightning.ai")
    return True


def get_studios():
    """Get list of available Studios"""
    success, stdout, _ = run_command("lightning studio ls")
    if success:
        return stdout
    return ""


def get_teamspaces():
    """Get list of available Teamspaces"""
    success, stdout, _ = run_command("lightning teamspace ls")
    if success:
        return stdout
    return ""


def update_scheduler_config(studio_name, teamspace):
    """Update the auto_restart_scheduler.py with user's configuration"""
    try:
        with open('auto_restart_scheduler.py', 'r') as f:
            content = f.read()
        
        # Replace STUDIO_NAME
        content = re.sub(
            r'STUDIO_NAME = ".*?"',
            f'STUDIO_NAME = "{studio_name}"',
            content
        )
        
        # Replace TEAMSPACE
        content = re.sub(
            r'TEAMSPACE = ".*?"',
            f'TEAMSPACE = "{teamspace}"',
            content
        )
        
        with open('auto_restart_scheduler.py', 'w') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        return False


def main():
    print("=" * 50)
    print("Lightning.ai Auto-Restart Scheduler Setup")
    print("=" * 50)
    print()
    
    # Step 1: Check Lightning CLI
    if not check_lightning_cli():
        sys.exit(1)
    
    # Step 2: Check authentication
    if not check_authentication():
        sys.exit(1)
    
    # Step 3: Get Studio name
    print("\n" + "=" * 50)
    print("Available Studios:")
    print("=" * 50)
    studios = get_studios()
    if studios:
        print(studios)
    else:
        print("(Unable to fetch Studios list)")
    
    print()
    studio_name = input("Enter your Studio name: ").strip()
    
    if not studio_name:
        print("❌ Studio name cannot be empty")
        sys.exit(1)
    
    # Step 4: Get Teamspace
    print("\n" + "=" * 50)
    print("Available Teamspaces:")
    print("=" * 50)
    teamspaces = get_teamspaces()
    if teamspaces:
        print(teamspaces)
    else:
        print("(Unable to fetch Teamspaces list)")
    
    print()
    print("Format: username/teamspace (e.g., john/default)")
    teamspace = input("Enter your Teamspace: ").strip()
    
    if not teamspace:
        print("❌ Teamspace cannot be empty")
        sys.exit(1)
    
    # Step 5: Confirm
    print("\n" + "=" * 50)
    print("Configuration Summary:")
    print("=" * 50)
    print(f"  Studio:    {studio_name}")
    print(f"  Teamspace: {teamspace}")
    print()
    
    confirm = input("Is this correct? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\nSetup cancelled")
        sys.exit(0)
    
    # Step 6: Update configuration
    print("\nUpdating auto_restart_scheduler.py...")
    if not update_scheduler_config(studio_name, teamspace):
        sys.exit(1)
    
    print("✓ Configuration updated successfully")
    
    # Step 7: Done
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print()
    print("To start the scheduler:")
    print("  python3 auto_restart_scheduler.py")
    print()
    print("To run in background:")
    print("  nohup python3 auto_restart_scheduler.py > scheduler.log 2>&1 &")
    print()
    print("To monitor:")
    print("  tail -f auto_restart.log")
    print()
    
    start_now = input("Start the scheduler now? (y/n): ").strip().lower()
    
    if start_now == 'y':
        print("\nStarting scheduler...")
        print("Press Ctrl+C to stop")
        print()
        
        # Import and run the scheduler
        try:
            import auto_restart_scheduler
            auto_restart_scheduler.main()
        except KeyboardInterrupt:
            print("\n\nScheduler stopped by user")
        except Exception as e:
            print(f"\n❌ Error starting scheduler: {e}")
            print("\nYou can start it manually with:")
            print("  python3 auto_restart_scheduler.py")
    else:
        print("\nYou can start it later with:")
        print("  python3 auto_restart_scheduler.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
