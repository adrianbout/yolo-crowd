"""
SmartChairCounter - Main Application Entry Point
Starts the FastAPI backend server
"""

import sys
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('smartchair.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    import uvicorn

    logger.info("="*60)
    logger.info("SmartChairCounter Application Starting...")
    logger.info("="*60)

    # Start FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        app_dir=str(backend_path)
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
