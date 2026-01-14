# Quick Start: Auto-Restart Scheduler

## What You Got

Three files for automatic Lightning.ai training restarts:

1. **`auto_restart_scheduler.py`** - Main scheduler (runs on your local machine)
2. **`setup_scheduler.py`** - Interactive setup (configures the scheduler)
3. **`AUTO_RESTART_SETUP.md`** - Detailed documentation

## How to Use

### One-Time Setup

On your **local machine** (not Lightning.ai):

```bash
# 1. Install Lightning SDK
pip install lightning-sdk

# 2. Login to Lightning.ai
lightning login

# 3. Run setup script
python3 setup_scheduler.py
```

The setup script will:
- Check if Lightning CLI is installed
- Verify authentication
- Show your Studios and Teamspaces
- Ask you to select which one to use
- Configure the scheduler
- Optionally start it immediately

### Running the Scheduler

After setup:

```bash
python3 auto_restart_scheduler.py
```

This will:
- Start your Lightning.ai Studio
- Run your training (`python3 main.py`)
- Wait 3.5 hours
- Restart the Studio
- Resume training from checkpoint
- Repeat forever

### Run in Background

```bash
nohup python3 auto_restart_scheduler.py > scheduler.log 2>&1 &
```

Monitor with:
```bash
tail -f auto_restart.log
```

## Requirements

- **Always-on machine** (local computer, VPS, Raspberry Pi, etc.)
- **Lightning.ai account** with Studio created
- **Your training code** uploaded to the Studio

## How It Works

```
Your Local Machine          Lightning.ai Cloud
─────────────────          ──────────────────
                                              
auto_restart_scheduler.py ──► Start Studio
                              Run training
                              (3.5 hours)
                              
                           ──► Stop Studio
                              Start Studio
                              Resume training
                              (3.5 hours)
                              
                           ──► Repeat...
```

Your checkpoints ensure training continues from where it left off!

## Cost

- **Lightning.ai:** Free tier ✓
- **Local machine:** Free (if you have one that stays on)
- **VPS (optional):** ~$5-6/month

## Troubleshooting

**"Lightning CLI not found"**
```bash
pip install lightning-sdk
```

**"Not authenticated"**
```bash
lightning login
```

**"Studio not found"**
- Run `lightning studio ls` to see available Studios
- Use exact Studio name in setup

## Alternative: Manual Restart

Don't want to run a scheduler? Just manually run this every 4 hours:

```bash
python3 main.py
```

The checkpoint system will still resume from where you left off!
