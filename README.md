# ML_Project_2

## Download the data

The ground truth lives in the Cascade repository under `Ground_truth/`, one
folder per dataset (`DS01-...`, `X-NAOMi-...`), each holding many small
`*_mini.mat` files (one per cell). `src/cascade_data/cli.py` lists and downloads
them into `data/raw/<dataset>/`.

```bash
# List the available datasets
python src/cascade_data/cli.py list

# List the files of one dataset
python src/cascade_data/cli.py list --files DS28-XCaMPgf-m-V1

# Download the smallest file of every dataset (default: --limit 1)
python src/cascade_data/cli.py download

# Download a few datasets, one file each
python src/cascade_data/cli.py download DS28-XCaMPgf-m-V1 DS29-GCaMP7f-m-V1 --limit 1

# Download ALL files of ALL datasets (large!)
python src/cascade_data/cli.py download --limit 1000000

# Preview what would be fetched, without downloading
python src/cascade_data/cli.py download --limit 1000000 --dry-run
```

Downloads are idempotent (existing files are skipped), so re-running is safe.
Files are ordered smallest-first, so `--limit N` grabs the `N` cheapest files
per dataset.

After downloading, build the local index (a JSON catalogue of the `.mat` files)
and inspect it:

```bash
python src/cascade_data/cli.py index --force   # (re)build data/index.json
python src/cascade_data/cli.py show            # list indexed sessions
python src/cascade_data/cli.py show --dataset DS28-XCaMPgf-m-V1
```

## Plot the raw sessions

Sanity-check the downloaded traces (dF/F, spikes, zoomed panel) before training:

```bash
python scripts/plot_sessions.py                          # every indexed session
python scripts/plot_sessions.py --dataset DS28-XCaMPgf-m-V1
python scripts/plot_sessions.py --limit 3                # first 3 files only
```

Figures are written to `plots/<dataset>/<session>.png`.