import sys
from pathlib import Path

# add project root to python path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import GPT2_SMALL_124M
# rest of your code...

print(GPT2_SMALL_124M.vocab_size)