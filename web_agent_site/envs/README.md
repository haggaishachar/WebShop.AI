# WebShop Environments

This directory contains the Gymnasium environment implementations for WebShop.

## ChromeDriver Setup

The `WebAgentSiteEnv` requires ChromeDriver to control a headless Chrome browser.

### Option 1: System ChromeDriver (Recommended)
Install ChromeDriver system-wide:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install chromium-chromedriver
```

**Other Linux:**
Download from https://chromedriver.chromium.org/downloads and place in your PATH

### Option 2: Local ChromeDriver
Download ChromeDriver for your system and place it at:
```
web_agent_site/envs/chromedriver
```

Make sure it matches your OS and architecture:
- Linux x86_64
- macOS (Intel or Apple Silicon)
- Windows

## Running the Site Environment

**Important:** The `WebAgentSiteEnv` requires the Flask webapp to be running first!

1. In one terminal, start the Flask app:
   ```bash
   make run-dev
   ```

2. In another terminal, run the site environment:
   ```bash
   make run-web-agent-site
   # or
   ./run_web_agent_site_env.sh
   ```

## Text Environment

The `WebAgentTextEnv` doesn't require ChromeDriver or a running Flask app. It simulates the environment internally.

```bash
make run-random-policy
# or  
./run_web_agent_text_env.sh
```

