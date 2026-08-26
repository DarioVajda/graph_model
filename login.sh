#!/bin/bash
# Logs in to Hugging Face and Weights & Biases.  Both helpers hold API tokens,
# so they are gitignored -- the repo ships .example templates instead.
cd "$(dirname "$0")" || exit 1

for script in hf_login.sh wandb_login.sh; do
  if [ ! -f "$script" ]; then
    echo "login.sh: $script is missing.  Set it up with:" >&2
    echo "    cp ${script%.sh}.example.sh $script" >&2
    echo "  then insert your API token into the copy." >&2
    exit 1
  fi
done

./hf_login.sh || exit 1
./wandb_login.sh || exit 1
