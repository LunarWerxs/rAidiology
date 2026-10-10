# Run rAidiology on your own computer

`launcher.py` serves the `site/` pages and a small control page at `/launcher/` on this computer only
(127.0.0.1). It uses only the Python standard library (3.9+) and never sends anything anywhere.

## The three files

Double-click the one for your system, in the folder you unzipped:

- `start-windows.bat`
- `start-mac.command`
- `start-linux.sh`

Each finds Python and runs `reading/local/launcher.py`. Your browser opens at the control page.

## Flags

- `--port N`: use port N (default 8765, or the next free port up to 8799)
- `--no-browser`: do not open the browser (also `RAIDIOLOGY_NO_BROWSER=1`)
- `--exit-after SECONDS`: stop by itself after that long

## Status

`GET /launcher/api/status` returns JSON: `version`, `python`, `os`, `gpu` (true if `nvidia-smi` exists; no
GPU code is run), `disk_free_bytes`, and `models` (from `models.json`, each with `installed`).
`POST /launcher/api/stop` stops it, only from this page.

## Weights

Research-model weights are gated downloads on Hugging Face. Put each in `models/<key>/` inside the project
folder (a `.safetensors`, `.onnx` or `.pt` file). Keys are listed in `models.json`.

Not a medical device. For education and research.
