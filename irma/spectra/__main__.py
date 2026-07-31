"""Allow running the spectra CLI as: python -m irma.spectra ..."""
import sys

from irma.spectra.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
