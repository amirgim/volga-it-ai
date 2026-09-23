"""Run: python -m src.selfcheck  -- fails if postprocess logic breaks."""
from .postprocess import is_valid, mask_low_conf, normalize


def main() -> None:
    assert normalize("а123вс777") == "A123BC777"
    assert is_valid("A123BC777")
    assert is_valid("A123BC77")
    assert is_valid("A123BC177")
    assert not is_valid("A123BC377")          # 3-digit region must start 1|2|7
    assert not is_valid("A123B777")           # missing second letter
    assert mask_low_conf("A123BC777", [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) == "A1#3BC777"
    print("selfcheck OK")


if __name__ == "__main__":
    main()
