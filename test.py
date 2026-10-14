import sys
from pathlib import Path
from tools.train_sfun import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(["--eval-only", "--checkpoint", str(root / "checkpoints/sfun.pth"),
          "--output", str(root / "outputs/test")] + sys.argv[1:])
