#!/bin/bash
# Template.  Copy to wandb_login.sh (gitignored) and replace the placeholder
# with your key from https://wandb.ai/authorize
TOKEN="INSERT YOUR API TOKEN HERE"

# --- remove this block once the token is inserted ---------------------------
# A real key is one unbroken word, so blank-or-contains-a-space means the
# placeholder is still there.
case "$TOKEN" in
  "" | *" "*)
    echo "wandb_login.sh: no token set -- edit the TOKEN line in this file." >&2
    exit 1
    ;;
esac
# ----------------------------------------------------------------------------

wandb login "$TOKEN"
