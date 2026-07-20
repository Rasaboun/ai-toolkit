"""
ai-toolkit on Modal.com — Gourmand LoRA training (Krea 2 capable)

Builds ai-toolkit from source (Rasaboun/ai-toolkit @ feat/mlflow-logging-modal-latest,
which is upstream ostris/ai-toolkit main merged 2026-07-20 + our Modal/MLflow patches)
on CUDA 12.8.1 + PyTorch 2.9.1 (pairing follows upstream docker/Dockerfile —
torchcodec 0.9.1 requires torch 2.9.x), then serves the web UI on
RTX PRO 6000 (96 GB VRAM).

Workflow:
  1. One-time: upload training dataset
       MODAL_PROFILE=<account> modal run run_modal.py

  2. Development (warm, URL printed to terminal):
       MODAL_PROFILE=<account> modal serve run_modal.py

  3. Production (persistent URL):
       MODAL_PROFILE=<account> modal deploy run_modal.py

NOTE: min_containers=0 → no idle GPU burn while nothing is training.
Before starting a long unattended training run, set min_containers=1 and
redeploy, otherwise Modal may scale the container down (killing training)
after ~1h without HTTP traffic.
"""

import os
import subprocess

import modal

output_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
dataset_volume = modal.Volume.from_name("aitk-dataset", create_if_missing=True)
db_volume = modal.Volume.from_name("aitk-db", create_if_missing=True)

OUTPUT_DIR = "/root/output-volume"
LOCAL_OUTPUT_DIR = "/app/ai-toolkit/output"
DATASET_DIR = "/app/ai-toolkit/datasets"
DB_DIR = "/root/aitk-db"
AITK_DIR = "/app/ai-toolkit"

AITK_REPO = "https://github.com/Rasaboun/ai-toolkit.git"
AITK_BRANCH = "feat/mlflow-logging-modal-latest"
AITK_COMMIT = "49871ac"  # ostris/main (Krea 2) merged into fork, 2026-07-20

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git", "curl", "build-essential", "cmake", "wget",
        "ffmpeg", "libgl1", "libglib2.0-0",
    )
    # torch BEFORE requirements so the CUDA wheel is authoritative
    # (versions mirror upstream docker/Dockerfile)
    .run_commands(
        "pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1"
        " --index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        f"git clone -b {AITK_BRANCH} {AITK_REPO} {AITK_DIR}"
        f" && cd {AITK_DIR} && git checkout {AITK_COMMIT}",
        f"rm -rf {AITK_DIR}/datasets",  # volume mounts here
    )
    .run_commands(
        f"pip install -r {AITK_DIR}/requirements.txt",
        "pip install setuptools==69.5.1",
        "pip install 'mlflow>=3,<4'",
    )
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_23.x | bash -",
        "apt-get install -y nodejs",
    )
    # Modal-proxy hardening (applied BEFORE the single npm build):
    # 1) guard against YAML.parse(null) crash in the advanced job editor
    # 2) files/log routes return empty payloads instead of 404 while a job
    #    isn't in the DB yet (UI polls them and breaks on 404 behind Modal's proxy)
    .run_commands(
        # each sed is followed by a grep assertion so a silent no-op fails the build
        'sed -i \'/let parsed = YAML.parse(value);/a\\      if (!parsed || typeof parsed !== "object") return;\' '
        f"{AITK_DIR}/ui/src/components/AdvancedConfigEditor.tsx"
        ' && grep -q \'typeof parsed !== "object"\' '
        f"{AITK_DIR}/ui/src/components/AdvancedConfigEditor.tsx",
        'sed -i "s|return NextResponse.json({ error: \'Job not found\' }, { status: 404 });|return NextResponse.json({ files: [] });|" '
        f"'{AITK_DIR}/ui/src/app/api/jobs/[jobID]/files/route.ts'"
        " && ! grep -q 'Job not found' "
        f"'{AITK_DIR}/ui/src/app/api/jobs/[jobID]/files/route.ts'",
        'sed -i "s|return NextResponse.json({ error: \'Job not found\' }, { status: 404 });|return NextResponse.json({ log: \'\' });|" '
        f"'{AITK_DIR}/ui/src/app/api/jobs/[jobID]/log/route.ts'"
        " && ! grep -q 'Job not found' "
        f"'{AITK_DIR}/ui/src/app/api/jobs/[jobID]/log/route.ts'",
    )
    .run_commands(
        f"cd {AITK_DIR}/ui && npm install && npm run update_db && npm run build"
    )
)

app = modal.App(
    name="aitk-gourmand",
    image=image,
    volumes={
        DB_DIR: db_volume,
        DATASET_DIR: dataset_volume,
        OUTPUT_DIR: output_volume,
    },
)

LOCAL_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "image-generator", "datasets", "v4"
)


@app.local_entrypoint()
def upload_dataset():
    """
    Upload the v4 food dataset (438 images + 438 captions) to the
    aitk-dataset volume.

    Usage:
      MODAL_PROFILE=<account> modal run run_modal.py
    """
    import pathlib

    src = pathlib.Path(LOCAL_DATASET_DIR).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found at {src}")
    files = list(src.iterdir())
    print(f"Uploading {len(files)} files from {src} ...")
    vol = modal.Volume.from_name("aitk-dataset", create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(src), "/v4")
    print(f"Done. {len(files)} files uploaded to aitk-dataset volume.")
    print(f"Dataset will be available at {DATASET_DIR}/v4 inside the container.")


# MLflow intentionally not configured for this deployment (out of scope).
# To re-enable: set MLFLOW_TRACKING_URI + MLFLOW_EXPERIMENT_NAME env vars in
# serve() below — the fork's CompositeLogger auto-activates when they exist.


@app.function(
    gpu="RTX-PRO-6000",
    timeout=28800,
    max_containers=1,
    min_containers=0,  # no idle burn; set to 1 before unattended training
    scaledown_window=3600,
)
@modal.web_server(8675, startup_timeout=120)
def serve():
    """
    Runs ai-toolkit UI on port 8675. Modal proxies HTTPS traffic to it.
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    dataset_volume.reload()
    output_volume.reload()
    db_volume.reload()

    import shutil
    import threading

    # --- restore persisted state from volumes -------------------------------
    db_path = f"{AITK_DIR}/aitk_db.db"
    persistent_db = f"{DB_DIR}/aitk_db.db"
    if os.path.exists(persistent_db):
        if os.path.islink(db_path):
            os.remove(db_path)
        shutil.copy2(persistent_db, db_path)
        print("DB: restored from volume")
    else:
        print("DB: fresh (first run)")

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        subprocess.run(["cp", "-a", f"{OUTPUT_DIR}/.", LOCAL_OUTPUT_DIR], check=False)
        print(f"Output: restored from volume ({len(os.listdir(LOCAL_OUTPUT_DIR))} entries)")
    else:
        print("Output: fresh local dir")

    # --- periodic backup local → volumes ------------------------------------
    def _backup_loop():
        """Sync local DB + output to volumes every 2 min."""
        import time

        while True:
            time.sleep(120)
            try:
                if os.path.exists(db_path):
                    shutil.copy2(db_path, persistent_db)
                    db_volume.commit()
                if os.path.exists(LOCAL_OUTPUT_DIR) and os.listdir(LOCAL_OUTPUT_DIR):
                    subprocess.run(
                        ["cp", "-a", f"{LOCAL_OUTPUT_DIR}/.", OUTPUT_DIR], check=False
                    )
                    output_volume.commit()
            except Exception as e:
                print(f"backup loop error: {e}")

    backup_thread = threading.Thread(target=_backup_loop, daemon=True)
    backup_thread.start()
    print("Backup thread started (every 2min → volumes)")

    if os.path.exists(DATASET_DIR):
        dataset_files = os.listdir(DATASET_DIR)
        print(f"Dataset: {DATASET_DIR} ({len(dataset_files)} entries)")
    else:
        print(f"Dataset: {DATASET_DIR} (not found — run: modal run run_modal.py)")
    print(f"Output:  {LOCAL_OUTPUT_DIR} (local) → {OUTPUT_DIR} (volume backup)")

    subprocess.Popen(["npm", "run", "start"], cwd=f"{AITK_DIR}/ui")
