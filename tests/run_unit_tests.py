"""Lightweight test runner for pykt-toolkit.

Discovers and executes all test_*.py files in this directory without requiring pytest.
Uses importlib to load each test module and executes callables whose names start with 'test_'.

Exit codes:
 0 - all tests passed
 1 - one or more tests failed

Usage:
  python tests/run_unit_tests.py
"""
import os
import sys
import importlib.util
import traceback
from types import ModuleType

TEST_PREFIX = 'test_'
BASE_DIR = os.path.dirname(__file__)


def discover_test_files():
    for fname in os.listdir(BASE_DIR):
        if fname.startswith(TEST_PREFIX) and fname.endswith('.py') and fname != os.path.basename(__file__):
            yield os.path.join(BASE_DIR, fname)


def load_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    return module


def run_tests_in_module(module: ModuleType):
    results = []
    for name in dir(module):
        if name.startswith('test_'):
            obj = getattr(module, name)
            if callable(obj):
                try:
                    obj()
                    results.append((name, True, None))
                except AssertionError as e:
                    results.append((name, False, f'AssertionError: {e}'))
                except Exception as e:  # noqa
                    tb = traceback.format_exc()
                    results.append((name, False, f'Exception: {e}\n{tb}'))
    return results


def main():
    all_results = []
    for test_file in discover_test_files():
        module = load_module(test_file)
        module_results = run_tests_in_module(module)
        all_results.extend([(os.path.basename(test_file),) + r for r in module_results])

    # Reporting
    passed = sum(1 for r in all_results if r[2])
    failed = [r for r in all_results if not r[2]]

    print("\nTest Results Summary")
    print("=====================")
    for file_name, test_name, ok, err in all_results:
        status = 'PASS' if ok else 'FAIL'
        print(f"{status:<5} {file_name}:{test_name}")
        if err:
            print(f"       -> {err}")

    print("\nTotals: PASS={} FAIL={}".format(passed, len(failed)))
    if failed:
        print("One or more tests failed.")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
