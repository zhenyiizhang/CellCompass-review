from __future__ import annotations

import tempfile
import textwrap
import time
import unittest
import os
from pathlib import Path

from cytobridge_agent.tools.code_executor import CodeExecutor


class CodeExecutorTimeoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_start_method = os.environ.get("CYTOBRIDGE_CODE_EXECUTOR_START_METHOD")
        self._old_home = os.environ.get("HOME")
        self._home_tmp = tempfile.TemporaryDirectory()
        os.environ["CYTOBRIDGE_CODE_EXECUTOR_START_METHOD"] = "fork"
        os.environ["HOME"] = self._home_tmp.name

    def tearDown(self) -> None:
        if self._old_start_method is None:
            os.environ.pop("CYTOBRIDGE_CODE_EXECUTOR_START_METHOD", None)
        else:
            os.environ["CYTOBRIDGE_CODE_EXECUTOR_START_METHOD"] = self._old_start_method
        if self._old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._old_home
        self._home_tmp.cleanup()

    def test_isolated_execute_python_times_out_by_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = CodeExecutor(adata=None, output_dir=Path(tmp), input_path=None)

            started = time.monotonic()
            result = executor.execute(
                "import time\nprint('started')\ntime.sleep(5)\nprint('finished')",
                timeout=1,
                timeout_mode="isolated",
            )
            elapsed = time.monotonic() - started

            self.assertFalse(result["success"], result)
            self.assertTrue(result["timed_out"], result)
            self.assertIn("Timed Out", result["error"])
            self.assertLess(elapsed, 4.0)

    def test_isolated_saved_script_times_out_by_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "slow_script.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import time
                    print('started')
                    time.sleep(5)
                    print('finished')
                    """
                ).strip()
            )
            executor = CodeExecutor(adata=None, output_dir=tmp_path, input_path=None)

            started = time.monotonic()
            result = executor.execute_script_file(
                str(script),
                timeout=1,
                timeout_mode="isolated",
            )
            elapsed = time.monotonic() - started

            self.assertFalse(result["success"], result)
            self.assertTrue(result["timed_out"], result)
            self.assertIn("timed out", result["error"])
            self.assertLess(elapsed, 4.0)


if __name__ == "__main__":
    unittest.main()
