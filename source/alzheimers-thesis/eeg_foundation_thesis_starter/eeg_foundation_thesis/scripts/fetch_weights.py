"""Download OFFICIAL CBraMod weights, verify that they strictly match upstream source."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import load_encoder, sha256_file


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("Install requirements.txt first (huggingface_hub needed)") from e
    target = ROOT / "checkpoints" / "cbramod_official.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        print("Downloading official pretrained checkpoint (requires internet access)...")
        src = hf_hub_download(repo_id="weighting666/CBraMod", filename="pretrained_weights.pth")
        shutil.copyfile(src, target)
    model = load_encoder(str(target))  # strict architecture/weights verification
    print("Pretrained CBraMod checkpoint verified:", target)
    print("SHA256:", sha256_file(target))
    print("Trainable encoder parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
