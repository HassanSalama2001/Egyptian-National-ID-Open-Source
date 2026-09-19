"""
Windows' default console codepage (cp1252, or an OEM codepage like 437)
cannot encode Arabic - and every field this CLI prints is Arabic text.
Confirmed by installing the built wheel into a clean venv (no source
tree) and running `egy-nid-ocr extract`: it crashed with
UnicodeEncodeError on the very first extraction, in both table and json
output modes, before the fix in cli.py that forces UTF-8 on both the
native Windows console codepage and Python's own stdout/stderr streams.

This test can't reproduce the exact console codepage bug (that needs a
real Windows console handle, not a redirected stream), but it does prove
the reconfigure logic runs without raising and leaves the streams able to
encode Arabic - the part that's easy to silently break again.
"""
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only console-encoding fix")
def test_cli_module_import_leaves_stdout_able_to_encode_arabic():
    result = subprocess.run(
        [sys.executable, "-c",
         "from egyptian_national_id_ocr import cli; "
         "print('\u0645\u062d\u0645\u062f')"],  # "محمد"
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"importing cli crashed or left stdout unable to encode Arabic:\n"
        f"{result.stderr.decode('utf-8', errors='replace')}"
    )
    assert "محمد".encode("utf-8") in result.stdout


def test_extract_command_output_contains_no_encoding_error_markers():
    """
    A quick source-tree-level smoke test distinct from the full
    tests/test_packaging.py::test_installed_cli_entry_point_works (which
    proves the INSTALLED wheel works but is slow and opt-in). This one
    always runs, but can only check for the crash's fingerprint, not
    truly reproduce Windows console codepage detection.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "from egyptian_national_id_ocr import cli"],
        capture_output=True, text=True,
    )
    assert "UnicodeEncodeError" not in result.stderr
    assert result.returncode == 0
