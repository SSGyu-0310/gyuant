import os
import subprocess
import sys
import time
from datetime import datetime

import schedule

from utils.logger import setup_logger

# Setup Logger
logger = setup_logger('scheduler', 'scheduler.log')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
US_MARKET_DIR = os.path.join(BASE_DIR, 'us_market')

def run_update():
    """Executes the main update script."""
    logger.info("Starting scheduled update...")
    try:
        update_script = os.path.join(US_MARKET_DIR, 'update_all.py')
        result = subprocess.run(
            [sys.executable, update_script],
            cwd=US_MARKET_DIR,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Update completed successfully.")
            logger.debug(f"Output: {result.stdout}")
        else:
            logger.error(f"Update failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to execute update script: {e}")

def run_ai_analysis():
    """Executes AI analysis scripts specifically."""
    logger.info("Starting scheduled AI analysis...")
    try:
        for script in ('macro_analyzer.py', 'ai_summary_generator.py'):
            subprocess.run(
                [sys.executable, os.path.join(US_MARKET_DIR, script)],
                cwd=US_MARKET_DIR,
                check=True
            )
        logger.info("AI Analysis completed.")
    except Exception as e:
        logger.error(f"AI Analysis failed: {e}")

# --- Schedule Configuration ---
# US Market closes at 4:00 PM ET (approx 6:00 AM KST next day)
# Schedule update for 6:30 AM KST
schedule.every().day.at("06:30").do(run_update)

# Run AI analysis slightly later
schedule.every().day.at("07:00").do(run_ai_analysis)

# Example: Run every 4 hours for intraday data if needed
# schedule.every(4).hours.do(run_intraday_update)

if __name__ == "__main__":
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    
    # Run once on startup to verify (Optional, can be commented out)
    # run_update()
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            break
        except Exception as e:
            logger.error(f"scheduler loop error: {e}")
            time.sleep(60)
