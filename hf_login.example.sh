#!/bin/bash
# Template.  Copy to hf_login.sh (gitignored) and replace the placeholder with
# your token from https://huggingface.co/settings/tokens
TOKEN="INSERT YOUR API TOKEN HERE"

# --- remove this block once the token is inserted ---------------------------
# A real token is one unbroken word, so blank-or-contains-a-space means the
# placeholder is still there.
case "$TOKEN" in
  "" | *" "*)
    echo "hf_login.sh: no token set -- edit the TOKEN line in this file." >&2
    exit 1
    ;;
esac
# ----------------------------------------------------------------------------

huggingface-cli login --token "$TOKEN"
