#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
echo "The first time, macOS may offer to install its developer tools to get Python. Say Install and run this again."
if ! command -v python3 >/dev/null 2>&1; then
  echo "rAidiology needs Python 3.9 or newer. Install it from https://www.python.org/downloads/"
  read -r -p "Press Enter to close. " _
  exit 1
fi
exec python3 reading/local/launcher.py "$@"
