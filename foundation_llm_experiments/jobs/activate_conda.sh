#!/bin/bash
# Activate $CONDA_ENV on CURC. The env prefix and conda.sh are often
# in different trees:
#   envs:     /projects/$USER/software/anaconda/envs   (campus .condarc)
#             /projects/$USER/miniconda3/envs
#   conda.sh: module load anaconda  ->  /curc/sw/anaconda3/...
#             or miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-mono_s2s}"

_env_prefix=""
for _d in \
    "/projects/${USER}/software/anaconda/envs" \
    "/projects/${USER}/miniconda3/envs"
do
    if [ -x "$_d/$CONDA_ENV/bin/python" ]; then
        _env_prefix="$_d/$CONDA_ENV"
        break
    fi
done

if [ -z "$_env_prefix" ]; then
    echo "ERROR: no python at .../envs/$CONDA_ENV/bin/python"
    echo "Looked in /projects/$USER/software/anaconda/envs and /projects/$USER/miniconda3/envs"
    exit 1
fi

# Compute nodes: campus Anaconda module (do not load this on login nodes).
module load anaconda 2>/dev/null || true

_activate_ok=0
if command -v conda >/dev/null 2>&1; then
    conda activate "$_env_prefix" && _activate_ok=1
fi

if [ "$_activate_ok" != 1 ]; then
    for _sh in \
        "/projects/${USER}/miniconda3/etc/profile.d/conda.sh" \
        /curc/sw/anaconda3/*/etc/profile.d/conda.sh
    do
        [ -f "$_sh" ] || continue
        # shellcheck source=/dev/null
        source "$_sh"
        conda activate "$_env_prefix" && _activate_ok=1 && break
    done
fi

if [ "$_activate_ok" != 1 ] && [ -x "$_env_prefix/bin/python" ]; then
    export PATH="$_env_prefix/bin:$PATH"
    _activate_ok=1
    echo "PATH-prepended $_env_prefix (conda activate unavailable)"
fi

if [ "$_activate_ok" != 1 ]; then
    echo "ERROR: could not activate $CONDA_ENV at $_env_prefix"
    exit 1
fi

echo "Activated $CONDA_ENV from $_env_prefix"
echo "python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || true
unset _d _sh _env_prefix _activate_ok
