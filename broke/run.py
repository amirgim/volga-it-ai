
import argparse
from broke.pipeline import run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Broke stack inference")
    ap.add_argument("--input", required=True,
                    help="directory with .jpg/.png images")
    ap.add_argument("--output", default="result.csv")
    args = ap.parse_args()
    run_dir(args.input, args.output)


if __name__ == "__main__":
    main()
