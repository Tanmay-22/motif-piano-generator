# Motif v2 training and deployment runbook

This is the complete handoff from the finished source code to the public v2
demo. GPU training and the GitHub/Render account actions cannot be performed by
the application itself; every other check is automated in this repository.

## 1. Publish the finished source code

Before opening Colab, commit and push the v2 source so the notebook clones the
same revision you intend to deploy.

```powershell
git status
git add .github DEPLOYMENT.md Dockerfile README.md render.yaml pyproject.toml requirements.txt requirements-colab.txt src tests training web scripts .env.example
git commit -m "Complete category-aware motif model v2 deployment workflow"
git push origin main
```

Do not add `metrics.json`, `*.pt`, `*.mid`, the MAESTRO archive, dataset
directories, or `artifacts/`. They are intentionally ignored or local-only.
Wait for the GitHub **CI** check to pass.

## 2. Train or resume on a free Colab GPU

1. Open
   [training/train_v2_colab.ipynb](https://colab.research.google.com/github/Tanmay-22/motif-piano-generator/blob/main/training/train_v2_colab.ipynb).
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run the Drive-mount, repository, and directory cells.
4. Run **Train or resume**. The first run downloads MAESTRO and creates the
   split caches. Checkpoints are written to
   `MyDrive/motif-piano-v2/checkpoints`.
5. A free runtime may disconnect before 5,000 optimizer steps. Reconnect, rerun
   the setup cells, and rerun **Train or resume**. `--resume auto` restores the
   model, optimizer, scheduler, scaler, RNG state, epoch, and batch position
   from `latest.pt`.
6. Repeat until step 30,000 or until early stopping reports that validation
   loss stopped improving. The deployable file is always
   `conditioned-v2-best.pt`, selected by the lowest validation loss—not the
   final training step.

Do not use the final release cells while training is still improving.

## 3. Run the release evaluation and examples

Run the remaining notebook cells in order:

1. **Final held-out evaluation** evaluates every batch in the official
   validation and test splits (`--evaluation-max-batches 0`), compares the
   model with freshly initialized weights, and writes `evaluation-v2.json`.
2. Confirm all three quality gates are `true`:
   - validation loss beats the untrained model;
   - validation loss is worse when the correct motifs are shuffled;
   - held-out test loss is worse when the correct motifs are shuffled.
3. If a gate fails, do not release. Resume training and run the full evaluation
   again. Use `--allow-failed-quality-gates` only for diagnosing a deliberately
   experimental checkpoint, never for the public portfolio model.
4. **Fixed held-out listening examples** exports one motif, real continuation,
   and generated continuation for each period category.
5. **Build the public release bundle** validates the checkpoint and report,
   computes checksums, creates `model-v2.0.0-release.zip`, and generates
   `render-env-v2.txt` with the exact deployment values.
6. Download `conditioned-v2-best.pt`, the ZIP, and `render-env-v2.txt` from the
   final notebook cell.

The example MIDIs are MAESTRO-derived. Keep them in the non-commercial GitHub
Release bundle; do not commit them to the source repository.

## 4. Create the GitHub model release

In GitHub:

1. Open **Releases → Draft a new release**.
2. Create tag `model-v2.0.0` from `main`.
3. Use title `Motif conditioned piano model v2.0.0`.
4. Use the generated `RELEASE_NOTES.md` from the release directory as the
   release description; the packager has already filled in the measured values.
5. Upload these assets before publishing:
   - `conditioned-v2-best.pt`
   - `model-v2.0.0-release.zip`
   - all nine `.mid` files inside the release directory's `examples` folder
6. Publish only after both uploads finish.

Equivalent GitHub CLI command:

```powershell
gh auth status
git tag -a model-v2.0.0 -m "Motif conditioned piano model v2.0.0"
git push origin model-v2.0.0

$motifReleaseAssets = @(
  "C:\path\to\conditioned-v2-best.pt",
  "C:\path\to\model-v2.0.0-release.zip"
) + (Get-ChildItem -LiteralPath "C:\path\to\model-v2.0.0-release\examples" -Filter "*.mid").FullName
gh release create model-v2.0.0 $motifReleaseAssets `
  --repo Tanmay-22/motif-piano-generator `
  --title "Motif conditioned piano model v2.0.0" `
  --notes-file "C:\path\to\model-v2.0.0-release\RELEASE_NOTES.md" `
  --verify-tag
```

The pinned checkpoint URL should be:

```text
https://github.com/Tanmay-22/motif-piano-generator/releases/download/model-v2.0.0/conditioned-v2-best.pt
```

The checkpoint download URL and SHA-256 also appear in `render-env-v2.txt` and
under `checkpoint.download_url` and `checkpoint.sha256` in
`release-manifest-v2.json`. Keep the tag immutable: publish a new tag for every
later checkpoint instead of replacing this asset.

## 5. Configure and deploy Render

For a new service, choose **New → Blueprint**, connect this GitHub repository,
and apply `render.yaml`. For the existing service, open its **Environment** page
and update these values manually (Render does not apply updated `sync: false`
variables to an existing Blueprint service). Copy the complete contents of the
generated `render-env-v2.txt`; it has this shape:

```text
MODEL_PATH=/app/artifacts/conditioned-v2-best.pt
MODEL_URL=https://github.com/Tanmay-22/motif-piano-generator/releases/download/model-v2.0.0/conditioned-v2-best.pt
MODEL_SHA256=<the 64-character value generated by the release packager>
GITHUB_REPOSITORY_URL=https://github.com/Tanmay-22/motif-piano-generator
PUBLIC_BASE_URL=https://motif-piano-generator.onrender.com
GENERATION_QUEUE_TIMEOUT_SECONDS=5
TORCH_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Then select **Manual Deploy → Clear build cache & deploy** once. The Docker
build downloads the exact release asset and verifies its SHA-256 value. Later
source pushes deploy automatically only after GitHub CI passes.

In the Render build log, confirm that the checkpoint download succeeds and no
checksum error appears. A checksum failure is a safety stop: correct the URL or
SHA instead of removing verification.

## 6. Verify the public deployment

First open:

```text
https://motif-piano-generator.onrender.com/health
```

It must include:

```json
{
  "status": "ok",
  "model_ready": true,
  "model_error": null,
  "version": "2.0.0",
  "model_version": "v2"
}
```

Run the automated end-to-end request from the repository:

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test_deployment `
  --base-url https://motif-piano-generator.onrender.com
```

Alternatively, open **Actions → Deployment smoke test → Run workflow** in
GitHub. The check wakes the free service, requests a five-second Romantic
continuation, validates the category/texture response, decodes the MIDI, and
confirms it contains continuation notes.

Finally test in the page:

1. The header says **Model v2 ready**.
2. The musical-direction selector is enabled.
3. Record both a single-note motif and a two-hand/chord motif.
4. Generate five seconds with Auto and one named category.
5. Confirm the result shows model, category, and inferred texture chips.
6. Play and download the result MIDI.
7. Upload the downloaded MIDI once to confirm upload validation still works.
8. Try two simultaneous browser requests and confirm the second receives the
   clear busy response instead of crashing the service.

## 7. Document the measured result

After deployment, update the README with the real validation loss, held-out
test loss, motif-dependency gaps, training step, release link, and links to the
fixed listening examples. Commit that documentation-only change and let CI and
Render deploy it.

Render's free web service currently has 0.1 CPU and 512 MB RAM and spins down
after 15 minutes without inbound traffic. Cold starts and slower five-second
generation are expected; this configuration is a portfolio demo, not a
production music service.

## Rollback

If v2 fails after deployment, set `MODEL_URL` and `MODEL_SHA256` back to the
previous immutable release, set the corresponding `MODEL_PATH`, and redeploy.
Do not replace assets under an existing tag because that invalidates the pinned
URL/checksum pair and makes the deployment irreproducible.
