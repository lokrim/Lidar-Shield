import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "synthetic"


def test_fixture_generator_reproduces_committed_bytes(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_synthetic_fixtures.py"),
            "--output",
            str(output),
        ],
        check=True,
    )

    expected_files = sorted(
        path.relative_to(FIXTURE_ROOT)
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    )
    actual_files = sorted(
        path.relative_to(output) for path in output.rglob("*") if path.is_file()
    )
    assert actual_files == expected_files
    for relative_path in expected_files:
        assert (output / relative_path).read_bytes() == (
            FIXTURE_ROOT / relative_path
        ).read_bytes()
