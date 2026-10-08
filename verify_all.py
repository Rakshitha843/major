"""
ONE-COMMAND FINAL VERIFICATION (Phase 15).

Run this after installing everything (see README.md "Setup") to check the whole project in
one go, the way a marker or a new contributor would want to before trusting the system:

    python scripts/verify_all.py

It does four things, in order, and stops describing each step only after it finishes:

  1. Confirms this is being run from the project root (so relative paths below are correct).
  2. Byte-compiles every .py file in the project -- this catches plain syntax errors
     immediately, before wasting time on tests that would fail for a trivial reason.
  3. Runs scripts/check_environment.py -- are the required packages installed and the
     MediaPipe model files present? (Use --no-camera here if you are on a machine with no
     webcam attached, e.g. testing over a remote desktop session; the flag is passed straight
     through.)
  4. Runs the full automated test suite (python -m unittest discover), the same one described
     in README.md "Test", and prints a short summary of how many tests ran, passed, or were
     skipped.

None of this switches your webcam on for more than a fraction of a second (step 3's camera
check, which can be skipped with --no-camera) and no video or images are ever saved -- see
"Privacy" in README.md.

This script does NOT replace the manual walkthrough in docs/testing_checklist.md. Automated
tests check the code paths with simulated cameras and faces; only a human, in front of a real
webcam, can confirm that the calibration beeps are audible, that the on-screen text is
clear, or that a real blink is actually counted. Run both before considering the project done.

Exit code is 0 if every step passed, 1 otherwise, so this can be used as a pass/fail gate
(for example in a personal pre-submission checklist).
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _run(cmd, cwd=PROJECT_ROOT):
    """Run a subprocess, streaming its output live, and return (ok, elapsed_seconds)."""
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.monotonic() - start
    return result.returncode == 0, elapsed


def step_confirm_project_root():
    _print_header("Step 1/4: Confirming project layout")
    required = ["backend", "computer_vision", "ml", "database", "utils", "tests", "frontend", "requirements.txt"]
    missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
    if missing:
        print(f"[FAIL] This does not look like the project root (missing: {', '.join(missing)}).")
        print(f"       Run this script as: python scripts/verify_all.py   (from {PROJECT_ROOT})")
        return False
    print(f"[ OK ] Project root looks correct: {PROJECT_ROOT}")
    return True


def step_compile_everything():
    _print_header("Step 2/4: Checking every .py file compiles (catches syntax errors early)")
    py_files = sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in PROJECT_ROOT.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
    )
    print(f"Compiling {len(py_files)} file(s)...")
    ok, elapsed = _run([sys.executable, "-m", "py_compile", *py_files])
    if ok:
        print(f"[ OK ] All {len(py_files)} files compiled cleanly ({elapsed:.1f}s).")
    else:
        print("[FAIL] At least one file has a syntax error -- see the traceback above.")
    return ok


def step_check_environment(extra_args):
    _print_header("Step 3/4: Checking your environment (packages, model files, camera)")
    ok, elapsed = _run([sys.executable, "scripts/check_environment.py", *extra_args])
    if ok:
        print(f"[ OK ] Environment check passed ({elapsed:.1f}s).")
    else:
        print("[FAIL] Environment check found a problem -- see above. Fix it before continuing;")
        print("       the automated tests below do not need a real webcam or model files, but the")
        print("       actual running application does.")
    return ok


def step_run_tests():
    _print_header("Step 4/4: Running the automated test suite (this can take a few minutes)")
    ok, elapsed = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."])
    if ok:
        print(f"[ OK ] Test suite passed ({elapsed:.1f}s). See the 'Ran N tests ... OK' line above for the count.")
    else:
        print("[FAIL] At least one test failed or errored -- see the traceback(s) above.")
        print("       If tests are 'skipped' rather than failing, that is expected when optional")
        print("       packages (fastapi, httpx, mediapipe) are not installed; see README.md Setup.")
    return ok


def main():
    args = sys.argv[1:]  # anything extra (e.g. --no-camera) is forwarded to check_environment.py
    steps = [
        ("Project layout", lambda: step_confirm_project_root()),
        ("Compile all files", lambda: step_compile_everything()),
        ("Environment check", lambda: step_check_environment(args)),
        ("Automated tests", lambda: step_run_tests()),
    ]

    results = []
    for name, step in steps:
        passed = step()
        results.append((name, passed))
        if not passed and name == "Project layout":
            break  # nothing else can run meaningfully from the wrong folder

    _print_header("SUMMARY")
    for name, passed in results:
        print(f"  {'[ OK ]' if passed else '[FAIL]'} {name}")

    all_ok = all(passed for _, passed in results)
    print()
    if all_ok:
        print("Everything passed. Next: work through docs/testing_checklist.md with a real webcam,")
        print("then run:  python -m uvicorn backend.main:app --reload")
    else:
        print("One or more steps failed -- scroll up to the matching section for the detail, fix it,")
        print("and run this script again. Fix problems in the order they appear: an environment")
        print("problem (step 3) usually explains why tests (step 4) also fail.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
