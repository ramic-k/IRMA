"""Allow running IRMA as: python -m irma"""
import sys

from irma.cli import main

if __name__ == "__main__":
    sys.exit(main())
