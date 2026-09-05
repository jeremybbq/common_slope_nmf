from datetime import datetime, timezone

from examples._run_output import create_run_output_dir


def test_create_run_output_dir_uses_timestamp_subdirectory(tmp_path):
    now = datetime(2026, 8, 17, 14, 5, 6, 123456, tzinfo=timezone.utc)

    output_dir = create_run_output_dir(tmp_path / "output", now=now)

    assert output_dir == (
        tmp_path / "output" / "2026-08-17_14-05-06-123456"
    )
    assert output_dir.is_dir()
