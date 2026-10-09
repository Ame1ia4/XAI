"""Execute current controlled notebooks; retain dated copies and progress checkpoints.

Usage: .venv/bin/python experiments/run_comparison.py [notebook.ipynb ...]
Results are saved under results/2026-10-09; the 8 October results are retained.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/2026-10-09"
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix="controlled-lime-run-"))
for name, value in {
    "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    "MPLCONFIGDIR": str(WORK / "matplotlib"), "IPYTHONDIR": str(WORK / "ipython"),
    "JUPYTER_RUNTIME_DIR": str(WORK / "jupyter"),
    "HF_HOME": str(ROOT / ".venv/tabfm_model_cache"),
    "HF_HUB_DISABLE_XET": "1",
    "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}.items(): os.environ[name] = value

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

MODELS = {"randomForest.ipynb":"RandomForest", "CovTypeLogRegLimeExp.ipynb":"LogisticRegression",
          "coverType_XGBoost_activity.ipynb":"XGBoost", "Covertype_TabFM_XAI.ipynb":"TabFM"}
manifest_path = OUT / "execution_manifest.json"
MANIFEST = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs":[]}
MANIFEST.update({
    "date":"2026-10-09", "experiment":"controlled_lime_protocol_v1", "python":sys.version,
    "platform":platform.platform(), "executable":sys.executable,
    "protocol_sha256":hashlib.sha256((ROOT / "experiments/controlled_lime_protocol.py").read_bytes()).hexdigest(),
    "packages":{p:importlib.metadata.version(p) for p in
        ["numpy","pandas","scipy","scikit-learn","lime","xgboost","torch","tabfm","safetensors","nbclient"]},
    "thread_environment":{k:os.environ[k] for k in
        ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]},
    "old_results":"../2026-10-08", "hf_cache":os.environ["HF_HOME"],
})

PREAMBLE = '''
import json, time, numpy as np
from pathlib import Path
import controlled_lime_protocol as protocol
np.random.seed(protocol.RANDOM_STATE)
_checkpoint = Path(CHECKPOINT_PATH)
_completed_explanations = 0
_original_benchmark = protocol.benchmark_lime
def _save_progress(event):
    global _completed_explanations
    with _checkpoint.open("a") as stream:
        stream.write(json.dumps(event) + "\\n")
    _completed_explanations += 1
    record = event["observation"]
    if record["repeat"] == protocol.REPEATS:
        print(f"{record['model']}: {_completed_explanations}/1575 explanations; "
              f"case {record['case_index']}, features {record['feature_count']}, "
              f"budget {record['num_samples']}", flush=True)
def _checkpointed_benchmark(*args, **kwargs):
    kwargs["progress_callback"] = _save_progress
    return _original_benchmark(*args, **kwargs)
protocol.benchmark_lime = _checkpointed_benchmark
'''

EXPORT = '''
import pandas as pd, json, hashlib
_snapshot = Path(OUTPUT_DIRECTORY)
controlled_raw.to_csv(_snapshot / f"{model_name.lower()}_controlled_lime_raw.csv", index=False)
controlled_summary.to_csv(_snapshot / f"{model_name.lower()}_controlled_lime_summary.csv", index=False)
pd.DataFrame([{"model":model_name, **predictive_metrics}]).to_csv(
    _snapshot / f"{model_name.lower()}_controlled_predictive_metrics.csv", index=False)
def _fingerprint(value):
    return hashlib.sha256(pd.util.hash_pandas_object(value,index=True).to_numpy().tobytes()).hexdigest()
_context = {
    "model":model_name,"train_rows":len(X_train),"test_rows":len(X_test),
    "input_features":X_train.shape[1],"background_rows":len(lime_background),
    "case_indices":explanation_cases.index.tolist(),"case_count":len(explanation_cases),
    "predictive_metrics":predictive_metrics,
    "data_fingerprints":{name:_fingerprint(globals()[name]) for name in
        ["X_train","X_test","y_train","y_test","lime_background","explanation_cases"]},
    "protocol":{name:getattr(protocol,name) for name in
        ["RANDOM_STATE","SAMPLE_SIZE","TEST_SIZE","BACKGROUND_SIZE","SAMPLE_BUDGETS","REPEATS",
         "EXPLANATIONS_PER_CLASS","NUM_FEATURES","TOP_K","FEATURE_COUNTS"]},
}
if model_name == "TabFM":
    _context["device"] = str(next(tabfm_model.parameters()).device)
    _context["torch_threads"] = torch.get_num_threads()
else: _context["device"] = "cpu"
(_snapshot / f"{model_name.lower()}_context.json").write_text(json.dumps(_context,indent=2))
'''

def save_manifest(): manifest_path.write_text(json.dumps(MANIFEST,indent=2))

def run_notebook(filename, name):
    path = ROOT / "experiments" / filename
    notebook = nbformat.read(path,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            if any(line.lstrip().startswith(("%pip install", "!pip install"))
                   for line in cell.source.splitlines()):
                cell.source = "# Setup skipped: dependencies are already installed in this kernel."
    checkpoint = OUT / f"{name.lower()}_checkpoint.jsonl"
    checkpoint.write_text("")
    notebook.cells.insert(0,nbformat.v4.new_code_cell(PREAMBLE.replace("CHECKPOINT_PATH",repr(str(checkpoint)))))
    notebook.cells.append(nbformat.v4.new_code_cell(EXPORT.replace("OUTPUT_DIRECTORY",repr(str(OUT)))))
    record = {"notebook":filename,"model":name,"status":"running",
        "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "execution_adaptations":["Dependency installs skipped; installed versions retained.",
            "Optional progress callback writes checkpoints outside the timed LIME call.",
            "CSV/context snapshot export appended; shared protocol values unchanged."],"cells":[]}
    previous = [r for r in MANIFEST["runs"] if r["notebook"] == filename]
    if previous:
        record["previous_attempts"] = previous
    MANIFEST["runs"] = [r for r in MANIFEST["runs"] if r["notebook"] != filename] + [record]
    save_manifest()
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook,km=km,timeout=14400,
        resources={"metadata":{"path":str(ROOT / "experiments")}})
    starts = {}
    def before(cell,cell_index,**kwargs):
        if cell.cell_type != "code": return
        starts[cell_index] = time.perf_counter()
        record["active_cell"] = cell_index
        save_manifest()
        print(f"{filename}: cell {cell_index} starting",flush=True)
    def after(cell,cell_index,**kwargs):
        if cell.cell_type != "code": return
        record["cells"].append({"index":cell_index,"seconds":time.perf_counter()-starts[cell_index]})
        nbformat.write(notebook,OUT / filename)
        save_manifest()
    client.on_cell_start = before
    client.on_cell_executed = after
    start = time.perf_counter()
    print(f"START {filename}",flush=True)
    try:
        client.execute()
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error))
        print(f"FAILED {filename}: {error}",flush=True)
    finally:
        record["notebook_wall_seconds"] = time.perf_counter()-start
        record["checkpoint_explanations"] = sum(1 for line in checkpoint.open() if line.strip())
        nbformat.write(notebook,OUT / filename)
        save_manifest()
        if client.kc is not None: client.kc.stop_channels()
        if km.has_kernel: km.shutdown_kernel(now=True)
    print(f"END {filename}: {record['status']} in {record['notebook_wall_seconds']:.1f}s",flush=True)

if __name__ == "__main__":
    backup = OUT / "before-rerun"
    backup.mkdir(exist_ok=True)
    for path in (ROOT / "experiments").glob("*_controlled_*.csv"):
        if not (backup / path.name).exists(): shutil.copy2(path,backup / path.name)
    requested = set(sys.argv[1:])
    for filename,name in MODELS.items():
        if not requested or filename in requested: run_notebook(filename,name)
    if len(MANIFEST["runs"]) == len(MODELS) and all(r["status"] != "running" for r in MANIFEST["runs"]):
        subprocess.run([sys.executable,str(ROOT / "experiments/summarize_comparison.py")],check=True)
