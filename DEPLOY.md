# Deployment Guide

Deploying to **Hugging Face Spaces** — free, built for Gradio, and handles
your large model files.

Total time: about 20 minutes, most of it waiting for uploads.

---

## STEP 1 — Add your model files

This zip contains the code. It does **not** contain your models or data
(those are ~800 MB and live in your Google Drive).

Download these from `MyDrive/PetProject` and put them in this folder:

| From Google Drive | Save into this folder as |
|---|---|
| `classes.json` | `classes.json` |
| `fish_classes.json` | `fish_classes.json` |
| `bird_classes.json` | `bird_classes.json` |
| `final_pet_model.pth` | `final_pet_model.pth` |
| `fish_model2.pth` *(or `fish_model.pth`)* | `fish_model.pth` |
| `bird_model.pth` | `bird_model.pth` |
| `pet_food_dataset (1).csv` | `pet_food_dataset.csv` |
| `pet_products.csv` | `pet_products.csv` |
| `food_bert_model/` *(whole folder)* | `food_bert_model/` |
| `productimages/` *(whole folder)* | `productimages/` |

**Two renames matter:**

1. `pet_food_dataset (1).csv` → `pet_food_dataset.csv`
   Spaces and brackets in filenames break on Linux servers.

2. Your notebook loads `fish_model2.pth` in the Gradio cell but
   `fish_model.pth` elsewhere. Use whichever produced the 71.43%
   accuracy, and name your copy `fish_model.pth`.

Delete the two `_PUT_..._HERE.txt` placeholder files once you have
copied the real contents in.

---

## STEP 2 — Verify before you upload

```bash
python check_files.py
```

This tells you exactly what is missing or misnamed. Do not continue
until it prints **"Everything is present. Ready to deploy."**

---

## STEP 3 — Test it locally

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860> and try all three tabs.

Fixing problems here takes seconds. Fixing them after upload means
another 800 MB push.

---

## STEP 4 — Create the Space

1. Sign up at <https://huggingface.co>
2. Click your avatar → **New Space**
3. Fill in:
   - **Space name:** `pet-breed-identifier`
   - **License:** MIT
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
4. Click **Create Space**. Leave the page open; you need the URL.

---

## STEP 5 — Push your files

Install Git LFS first if you do not have it: <https://git-lfs.com>

```bash
git lfs install

git clone https://huggingface.co/spaces/YOUR_USERNAME/pet-breed-identifier
cd pet-breed-identifier
```

Now copy **everything from this folder** into the cloned folder, then:

```bash
git lfs track "*.pth" "*.bin" "*.safetensors"
git add .gitattributes
git add .
git commit -m "Pet breed identification and product recommender"
git push
```

When prompted for a password, paste an access token — generate one at
**Settings → Access Tokens** with the **write** role. Your account
password will not work.

> **Run `git lfs install` before copying the `.pth` files in.**
> Without LFS the push is rejected: Hugging Face refuses ordinary Git
> files over 10 MB, and your models are ~110 MB each.

The upload takes 10–20 minutes on a normal connection.

---

## STEP 6 — Watch it build

Go to your Space page. It shows *Building* for 5–10 minutes, then
*Running*.

If it fails, open the **Logs** tab — the error is almost always a
missing file, which `check_files.py` would have caught in Step 2.

Your live app:
`https://huggingface.co/spaces/YOUR_USERNAME/pet-breed-identifier`

---

## What to expect once it is live

**First load takes 30–60 seconds.** Three Swin models plus BERT have to
load into memory. This is normal.

**Free Spaces sleep after 48 hours idle.** The next visitor triggers a
cold start of a minute or so. Fine for a demo or a project submission.

**Predictions take a few seconds.** You are on CPU now, not Colab's GPU.

**The bird head may be unreliable.** As noted in your paper, it was
trained without normalization but is served with it. Dog, cat and fish
predictions are unaffected.

---

## Updating later

```bash
# edit files, then
git add .
git commit -m "describe your change"
git push
```

The Space rebuilds automatically.

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` in logs | A file was not uploaded | Run `check_files.py`, add it, push again |
| Push rejected, "file too large" | LFS not set up | `git lfs install`, then `git lfs track "*.pth"`, recommit |
| `size mismatch` loading a model | Wrong `.pth` for that class count | Check the JSON class count matches the model head |
| Build fails on install | Dependency conflict | Read the Logs tab; pin the failing package version |
| App loads but images are blank | `productimages/` empty or names differ | Confirm filenames match the CSV columns |
