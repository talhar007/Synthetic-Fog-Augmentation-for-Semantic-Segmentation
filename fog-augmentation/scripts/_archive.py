"""
Shared helper for every results/figure-generating script: never let a
re-run silently clobber a previous output. Before writing to `path`, if a
file is already there, move it into `<path.parent>/previous_results/<timestamp>/`
so old tables/figures/checkpoints stay inspectable instead of being lost.

Usage:
    from _archive import archive_before_write
    archive_before_write(out_path)
    plt.savefig(out_path, ...)   # or open(out_path, "w") / csv writer / etc.
"""
from datetime import datetime
from pathlib import Path
import shutil


def archive_before_write(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = path.parent / "previous_results" / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(archive_dir / path.name))
    print(f"Archived previous → {archive_dir / path.name}")


if __name__ == "__main__":
    # CLI use: archive a file (e.g. a Generated_results/ figure about to be
    # `cp`-overwritten by hand) before it's replaced.
    import sys
    for p in sys.argv[1:]:
        archive_before_write(Path(p))
