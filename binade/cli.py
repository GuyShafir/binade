"""Command line: `binade pack ...` and `binade unpack ...`."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pack", "unpack"):
        sys.exit("usage: binade {pack,unpack} ...  (binade pack --help for options)")
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"binade {cmd}"
    if cmd == "pack":
        from .pack import main as run
    else:
        from .unpack import main as run
    run()
