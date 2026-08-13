from importlib import util
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def _run_example(script_name: str) -> None:
    script_path = EXAMPLES_DIR / script_name
    module_name = f"smoke_{script_path.stem}"
    spec = util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load example script: {script_path}")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


@pytest.mark.parametrize(
    "script_name",
    [
        "decision_tree_demo.py",
        "k_means_demo.py",
        "linear_regression_demo.py",
        "logistic_regression_demo.py",
        "pca_demo.py",
        "random_forest_demo.py",
    ],
)
def test_example_scripts_smoke(script_name: str):
    _run_example(script_name)
