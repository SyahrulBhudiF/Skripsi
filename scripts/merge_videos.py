from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

DEFAULT_INPUT_DIRS = [
    Path(
        "/home/ryuko/skripsi/Skripsi/Convat-1st/dataset/datatest/labeled/17_04_2026/"
        "anxiety_rendah/ilham_dharma_atmaja_1776397903000"
    ),
    Path(
        "/home/ryuko/skripsi/Skripsi/Convat-1st/dataset/datatest/labeled/17_04_2026/"
        "anxiety_rendah/muhammad_christiano_olyvian_barini_1776394401685"
    ),
    Path(
        "/home/ryuko/skripsi/Skripsi/Convat-1st/dataset/datatest/labeled/15_04_2026/"
        "anxiety_rendah/zukhrufian_abdullah_1776228449628"
    ),
]


def shell_quote_for_ffconcat(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "'\\''") + "'"


def collect_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def merge_videos(input_dir: Path, output: Path, reencode: bool) -> None:
    videos = collect_videos(input_dir)
    if not videos:
        raise SystemExit(f"Tidak ada video di: {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as file:
        list_path = Path(file.name)
        for video in videos:
            file.write(f"file {shell_quote_for_ffconcat(video)}\n")

    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
        if reencode:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an"]
        else:
            cmd += ["-c", "copy", "-an"]
        cmd.append(str(output))

        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)

    print(f"OK: {len(videos)} video -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gabungkan semua video dalam folder jadi 1 video tanpa audio.")
    parser.add_argument("input_dirs", nargs="*", type=Path, default=DEFAULT_INPUT_DIRS)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--reencode", action="store_true", help="Pakai ini kalau -c copy gagal karena codec/resolusi beda.")
    args = parser.parse_args()

    for input_dir in args.input_dirs:
        output = args.output or input_dir / "merged_no_audio.mp4"
        merge_videos(input_dir, output, args.reencode)


if __name__ == "__main__":
    main()
