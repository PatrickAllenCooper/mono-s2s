#!/bin/bash
# Find and activate $CONDA_ENV. CURC has two conda roots:
#   /projects/$USER/miniconda3              -> mono_s2s (CUDA 11.8, A100)
#   /projects/$USER/software/anaconda       -> mono_s2s_cu128 (CUDA 12.8, H200 / RTX Pro)
CONDA_ENV="${CONDA_ENV:-mono_s2s}"
_activate_ok=0
for _base in \
    ${CONDA_BASE:+"$CONDA_BASE"} \
    "/projects/${USER}/software/anaconda" \
    "/projects/${USER}/miniconda3"
do
    if [ -f "$_base/etc/profile.d/conda.sh" ] && [ -d "$_base/envs/$CONDA_ENV" ]; then
        # shellcheck source=/dev/null
        source "$_base/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" || continue
        CONDA_BASE="$_base"
        echo "Activated $CONDA_ENV from $CONDA_BASE"
        _activate_ok=1
        break
    fi
done
if [ "$_activate_ok" != 1 ]; then
    echo "ERROR: could not activate conda env '$CONDA_ENV'"
    echo "Looked in /projects/$USER/software/anaconda and /projects/$USER/miniconda3"
    exit 1
fi
unset _base _activate_ok
