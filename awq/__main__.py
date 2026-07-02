"""Allow running the package as: python -m awq"""
import sys

from awq.cli import main

if __name__ == "__main__":
    sys.exit(main())
