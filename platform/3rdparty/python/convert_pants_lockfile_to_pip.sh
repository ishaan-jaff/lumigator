#!/usr/bin/env bash

# converts a lockfile generated by pants to a pip-installable requirements.txt file.
LOCKFILE=$1

sed '/^\/\//d' "$LOCKFILE" >requirements.json
wheels="WHEELS_DIR|${WORKSPACE}/wheelhouse"
pex3 lock export --path-mapping $wheels requirements.json >requirements.txt
rm requirements.json

echo "to install the requirements file, run:"
echo "pip install --require-hashes --no-deps --no-cache-dir --upgrade -r requirements.txt"
