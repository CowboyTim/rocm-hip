#!/bin/sh
# store_verify.sh <model.gguf> [store_dir]: the KV block store across processes (tests/store_verify.cpp's phases).
set -e
BIN=$(dirname "$0")/../build/store_verify
MODEL=$1
DIR=${2:-/dev/shm/store_verify.$$}
rm -rf "$DIR" "$DIR.expected"
mkdir -p "$DIR"
"$BIN" "$MODEL" "$DIR" 1
"$BIN" "$MODEL" "$DIR" 2
BLOCK1=$("$BIN" "$MODEL" "$DIR" 0 | sed -n 's/^BLOCK1 //p')
rm -f "$DIR/$BLOCK1"
echo "  (deleted block 1: $BLOCK1)"
"$BIN" "$MODEL" "$DIR" 3
rm -rf "$DIR" "$DIR.expected"
