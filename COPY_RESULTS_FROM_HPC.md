# Copy Results from HPC to Local Workspace

## Method 1: SCP from Local Machine (Recommended)

Run these commands from **your local Windows machine** (PowerShell or Git Bash):

```bash
# Navigate to your project
cd C:\Users\patri\code\mono-s2s

# Create local results directory
mkdir -p downloaded_results

# IMPORTANT: Results are in seed-specific subdirectory!
# Copy all result JSON files for seed 42
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/seed_42/*.json downloaded_results/seed_42/

# Copy all CSV files
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/*.csv downloaded_results/

# Copy text summary
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/*.txt downloaded_results/

# Copy experiment summary
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/experiment_summary.txt downloaded_results/
```

## Method 2: SCP Specific Files

**IMPORTANT**: Results are saved in seed-specific subdirectories!
For seed 42: `/scratch/alpine/paco0228/mono_s2s_results/seed_42/`

If you only want specific files:

```bash
cd C:\Users\patri\code\mono-s2s

# Evaluation results (has monotonic ROUGE scores)
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/seed_42/evaluation_results.json downloaded_results/seed_42/

# Final aggregated results
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/final_results.json downloaded_results/

# Training histories
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/baseline_training_history.json downloaded_results/
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/monotonic_training_history.json downloaded_results/

# Attack results
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/hotflip_results.json downloaded_results/
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/uat_results.json downloaded_results/
```

## Method 3: Copy Entire Seed Directory (RECOMMENDED)

**This is the easiest and most reliable method**:

```bash
cd C:\Users\patri\code\mono-s2s

# Copy entire seed_42 directory
scp -r paco0228@login.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/seed_42 downloaded_results/

# For other seeds:
# scp -r paco0228@login.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/seed_1337 downloaded_results/
```

**Note**: Results are organized by seed in subdirectories (seed_42, seed_1337, etc.)

## Method 4: Using rsync (If Available)

```bash
cd C:\Users\patri\code\mono-s2s

# Rsync is more efficient for multiple files
rsync -avz paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/ downloaded_results/
```

## After Copying

### View the Results

```powershell
# View evaluation results
Get-Content downloaded_results\evaluation_results.json | ConvertFrom-Json | ConvertTo-Json

# Or use Python to view
python -c "import json; print(json.dumps(json.load(open('downloaded_results/evaluation_results.json')), indent=2))"

# Extract monotonic ROUGE scores
python -c "
import json
data = json.load(open('downloaded_results/evaluation_results.json'))
mono = data['cnn_dm']['monotonic_t5']['rouge_scores']
print('Monotonic ROUGE Scores:')
print(f\"  ROUGE-1: {mono['rouge1']['mean']:.4f} [{mono['rouge1']['lower']:.4f}, {mono['rouge1']['upper']:.4f}]\")
print(f\"  ROUGE-2: {mono['rouge2']['mean']:.4f} [{mono['rouge2']['lower']:.4f}, {mono['rouge2']['upper']:.4f}]\")
print(f\"  ROUGE-L: {mono['rougeLsum']['mean']:.4f} [{mono['rougeLsum']['lower']:.4f}, {mono['rougeLsum']['upper']:.4f}]\")
"
```

### Copy to mono_s2s_results (Keep in Repo)

```bash
# Copy to the tracked results directory
cp downloaded_results/*.json mono_s2s_results/
cp downloaded_results/*.txt mono_s2s_results/
cp downloaded_results/*.csv mono_s2s_results/

# Commit to git
git add mono_s2s_results/
git commit -m "Add verified experimental results from HPC (seed 42, full test set)"
git push origin main
```

## Quick Single Command

**Fastest way to get just evaluation results**:

```bash
scp paco0228@alpine.rc.colorado.edu:/scratch/alpine/paco0228/mono_s2s_results/evaluation_results.json C:\Users\patri\code\mono-s2s\
```

Then view:

```bash
python -c "import json; data=json.load(open('evaluation_results.json')); mono=data['cnn_dm']['monotonic_t5']['rouge_scores']; print(f\"Monotonic ROUGE-L: {mono['rougeLsum']['mean']:.4f}\")"
```

## Alternative: Copy from Documents Folder

If you already have results in `C:\Users\patri\Documents\mono-s2s`:

```powershell
# Copy from your Documents folder
Copy-Item "C:\Users\patri\Documents\mono-s2s\mono_s2s_results\*" `
          "C:\Users\patri\code\mono-s2s\downloaded_results\" -Recurse
```

---

**Run the scp commands above from your local terminal**, then tell me what you see in `evaluation_results.json` and I'll update the paper immediately!
