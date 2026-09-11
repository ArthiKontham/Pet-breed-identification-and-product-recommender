"""
Pet Breed Identification and Product Recommender — Streamlit app.

Run locally:   streamlit run app.py
Deployed on:   Streamlit Community Cloud
"""

import json
import os
import random

import pandas as pd
import streamlit as st
import torch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

# Optional fallback: if weights are missing locally (or GitHub LFS bandwidth is
# exhausted) they are pulled from this Hugging Face model repo instead.
# Leave as None to disable.
HF_REPO = None      # e.g. "arthi02/pet-breed-models"


def local(*parts):
    return os.path.join(BASE, *parts)


st.set_page_config(
    page_title="Pet Breed Identification and Product Recommender",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- Gradio-style theme ---- */
    .block-container { padding-top: 2rem; max-width: 1180px; }

    html, body, [class*="css"] { font-size: 1.02rem; }

    /* title + subtitle */
    #app-title {
        font-size: 1.95rem; font-weight: 700; color: #1f2937;
        margin: 0 0 .35rem 0; letter-spacing: -0.01em;
    }
    #app-subtitle { font-size: 1.05rem; color: #4b5563; margin: 0 0 1.1rem 0; }

    /* tabs: underline style with orange active state */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.8rem; border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem; font-weight: 500; color: #374151;
        padding: .55rem 0; background: transparent;
    }
    .stTabs [aria-selected="true"] { color: #f97316 !important; font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #f97316; height: 2px; }

    /* panels */
    .panel {
        border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1rem 1.1rem; background: #fff;
    }

    /* field labels */
    label, .stSelectbox label, .stTextInput label, .stTextArea label,
    .stNumberInput label, .stFileUploader label, .stRadio label {
        font-size: 1.02rem !important; font-weight: 500 !important; color: #374151 !important;
    }

    /* inputs */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    div[data-baseweb="select"] > div {
        border-radius: 7px !important; border-color: #d1d5db !important;
        font-size: 1.02rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #f97316 !important; box-shadow: 0 0 0 1px #f97316 !important;
    }

    /* primary button: solid orange, full width, like Gradio */
    .stButton > button {
        background: #f97316 !important; color: #fff !important;
        border: none !important; border-radius: 8px !important;
        font-size: 1.05rem !important; font-weight: 600 !important;
        padding: .65rem 1rem !important; width: 100%;
    }
    .stButton > button:hover { background: #ea6a0c !important; }

    /* radio row */
    .stRadio [role="radiogroup"] { gap: .6rem; }

    /* result card */
    .resultcard {
        border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1rem 1.2rem; background: #fff;
    }
    .resultcard .sub   { color: #6b7280; font-size: .92rem; margin-bottom: .25rem; }
    .resultcard .label { font-size: 1.5rem; font-weight: 700; color: #1f2937; }
    .pill {
        display: inline-block; margin-top: .45rem; padding: 3px 11px;
        border-radius: 999px; font-size: .85rem; font-weight: 600;
        background: #fff1e6; color: #c2540a;
    }

    .stMarkdown p { font-size: 1.02rem; }
    div[data-testid="stFileUploaderDropzone"] { border-radius: 8px; }
    [data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small files loaded from the repo
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_classes(filename):
    with open(local(filename), encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):                       # {"0": "beagle", ...}
        return [data[k] for k in sorted(data, key=lambda x: int(x))]
    return list(data)


@st.cache_data(show_spinner=False)
def load_csv(filename):
    return pd.read_csv(local(filename))


def is_lfs_pointer(path):
    """A file that git-lfs never resolved is a small text stub."""
    try:
        if os.path.getsize(path) > 1000:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def resolve(filename):
    """Return a usable path for a weights file, downloading it if necessary."""
    p = local(filename)
    if os.path.exists(p) and not is_lfs_pointer(p):
        return p
    if HF_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO, filename=filename)
    raise FileNotFoundError(
        f"'{filename}' is missing or is an unresolved Git LFS pointer. "
        "Either the file was not pushed, or the LFS bandwidth quota is exhausted. "
        "Set HF_REPO near the top of app.py to fall back to the Hugging Face Hub."
    )


# --------------------------------------------------------------------------
# Models — loaded only when their tab is used, and cached afterwards
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@st.cache_resource(show_spinner=False)
def get_vision_model(weights_file, num_classes):
    import timm
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    state = torch.load(resolve(weights_file), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner=False)
def get_food_model():
    from transformers import BertForSequenceClassification, BertTokenizer
    folder = local("food_bert_model")
    if not os.path.isdir(folder):
        raise FileNotFoundError("The 'food_bert_model' folder is missing from the repo.")
    tokenizer = BertTokenizer.from_pretrained(folder)
    model = BertForSequenceClassification.from_pretrained(folder)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def get_recommender():
    from sentence_transformers import SentenceTransformer
    products = load_csv("pet_products.csv").copy()
    for col in ("product_name", "pet_type", "pet_category"):
        products[col] = products[col].astype(str)
    products["text"] = (products["product_name"] + " "
                        + products["pet_type"] + " "
                        + products["pet_category"])
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(products["text"].tolist())
    return encoder, embeddings, products


HEADS = {
    "Dog / Cat": ("final_pet_model.pth", "classes.json"),
    "Fish":      ("fish_model.pth",      "fish_classes.json"),
    "Bird":      ("bird_model.pth",      "bird_classes.json"),
}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### About")
    st.write(
        "A multimodal pet-care assistant combining computer vision and "
        "natural language processing."
    )
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown(
        "- **Swin Transformer** — breed and species recognition\n"
        "- **BERT** — food ingredient safety\n"
        "- **Sentence-BERT** — product retrieval"
    )
    st.markdown("---")
    st.caption(
        "Models load the first time you use a tab, so the first result "
        "takes longer than the rest."
    )

st.markdown('<div id="app-title">Pet Breed Identification and Product Recommender</div>',
            unsafe_allow_html=True)
st.markdown('<div id="app-subtitle">Identify your pet, check whether a food is suitable, '
            'and get product suggestions.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["Breed Identification", "Product Recommendation", "Food Quality Check"]
)

# ---------------------------- Breed identification ------------------------

with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload a photo of your pet",
            type=["jpg", "jpeg", "png", "webp"],
        )
        animal_type = st.radio(
            "Animal type", list(HEADS.keys()), horizontal=True,
            help="Choosing the right group sends the image to the matching model.",
        )
        identify = st.button("Identify", type="primary", use_container_width=True)

    with right:
        if uploaded is not None:
            st.image(uploaded, caption="Uploaded photo", use_container_width=True)
        else:
            st.info("Upload a clear, well-lit photo where the animal fills most of the frame.")

    if identify:
        if uploaded is None:
            st.warning("Please upload an image first.")
        else:
            weights_file, classes_file = HEADS[animal_type]
            try:
                classes = load_classes(classes_file)
                with st.spinner("Loading the model (first run only)..."):
                    model = get_vision_model(weights_file, len(classes))

                image = Image.open(uploaded).convert("RGB")
                tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)

                k = min(3, probs.shape[1])
                top = torch.topk(probs, k=k)

                name = classes[top.indices[0][0].item()].replace("_", " ")
                if name == "Tzu":
                    name = "Shih Tzu"
                conf = top.values[0][0].item() * 100

                st.markdown(
                    f"""<div class="resultcard">
                          <div class="sub">Predicted {'breed' if animal_type == 'Dog / Cat' else 'species'}</div>
                          <div class="label">{name}</div>
                          <span class="pill">{conf:.2f}% confidence</span>
                        </div>""",
                    unsafe_allow_html=True,
                )
                st.progress(min(conf / 100, 1.0))

                if conf < 60:
                    st.info(
                        "Confidence is low. A closer, sharper photo with the animal "
                        "facing the camera usually improves the result."
                    )

                if k > 1:
                    with st.expander("Other possibilities"):
                        for score, idx in zip(top.values[0][1:], top.indices[0][1:]):
                            alt = classes[idx.item()].replace("_", " ")
                            st.write(f"**{alt}** — {score.item() * 100:.2f}%")

            except Exception as e:
                st.error(f"Could not run the model.\n\n{e}")

# ---------------------------- Product recommendation ----------------------

with tab2:
    left, right = st.columns([1, 2], gap="large")

    with left:
        p_type = st.selectbox("Pet type", ["dog", "cat", "bird", "fish"])
        p_breed = st.text_input("Breed / species", placeholder="e.g. Maine Coon")
        p_size = st.selectbox("Size / life stage", ["small", "big", "kitten", "adult"])
        c1, c2 = st.columns(2)
        p_age = c1.number_input("Age (years)", min_value=0, value=1)
        p_weight = c2.number_input("Weight (kg)", min_value=0, value=5)
        recommend = st.button("Recommend", type="primary", use_container_width=True)

    with right:
        if not recommend:
            st.info("Fill in your pet's details to see suggested products.")
        else:
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                with st.spinner("Loading the recommender (first run only)..."):
                    encoder, embeddings, products = get_recommender()

                size = "large" if p_size == "big" else p_size
                query = f"{p_type} {p_breed} {size} pet products"
                scores = cosine_similarity(encoder.encode([query]), embeddings)[0]
                top = scores.argsort()[-6:][::-1]

                st.markdown("**Recommended for your pet**")
                cols = st.columns(3)
                for n, idx in enumerate(top):
                    row = products.iloc[idx]
                    img_path = None
                    for c in ("image1", "image2", "image3"):
                        if c in products.columns and isinstance(row.get(c), str):
                            cand = local("productimages", os.path.basename(row[c]))
                            if os.path.exists(cand):
                                img_path = cand
                                break
                    with cols[n % 3]:
                        if img_path:
                            st.image(img_path, use_container_width=True)
                        st.caption(f"**{row['product_name']}**")
                        st.caption(f"match {scores[idx]*100:.0f}%")

            except Exception as e:
                st.error(f"Could not generate recommendations.\n\n{e}")

# ---------------------------- Food quality check --------------------------

with tab3:
    left, right = st.columns(2, gap="large")

    with left:
        f_type = st.selectbox("Pet type", ["dog", "cat", "bird", "fish"], key="ft")
        f_size = st.selectbox("Size / life stage",
                              ["small", "big", "kitten", "adult"], key="fs")
        f_text = st.text_area(
            "Food ingredients",
            placeholder="e.g. chicken meal, brown rice, carrot",
            height=110,
        )
        check = st.button("Check", type="primary", use_container_width=True)

    with right:
        if not check:
            st.info("Enter the ingredients printed on the packet, separated by commas.")
        elif not f_text.strip():
            st.warning("Please enter some ingredients.")
        else:
            try:
                food_df = load_csv("pet_food_dataset.csv")
                food = f_text.lower()

                subset = food_df[
                    (food_df["pet_type"].astype(str).str.lower() == f_type)
                    & (food_df["pet_category"].astype(str).str.lower() == f_size)
                ]

                verdict, source = None, "reference table"
                for _, row in subset.iterrows():
                    if str(row["food_name"]).lower() in food:
                        verdict = str(row["label"]).strip().lower()
                        break

                if verdict is None:
                    with st.spinner("Loading the classifier (first run only)..."):
                        tokenizer, model = get_food_model()
                    inputs = tokenizer(f"{f_type} {f_size} {food}",
                                       return_tensors="pt",
                                       truncation=True, padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    verdict = ["safe", "moderate", "unsafe"][torch.argmax(logits).item()]
                    source = "model prediction"

                if verdict == "safe":
                    st.success("### ✅ Safe\nThis food is suitable for your pet.")
                    st.write("Other good options: chicken, salmon, pumpkin.")
                elif verdict == "moderate":
                    st.warning("### ⚠️ Moderate\nGive this only occasionally.")
                    st.write("Large or frequent servings may cause problems.")
                else:
                    st.error("### ❌ Unsafe\nThis food may harm your pet.")
                    safe = subset[
                        subset["label"].astype(str).str.strip().str.lower() == "safe"
                    ]["food_name"].tolist()
                    if len(safe) < 4:
                        pool = food_df[
                            food_df["label"].astype(str).str.strip().str.lower() == "safe"
                        ]["food_name"]
                        safe = pool.sample(min(4, len(pool))).tolist()
                    if safe:
                        st.write("**Safer choices:** " + ", ".join(str(s) for s in safe[:4]))

                st.caption(f"Source: {source}")

            except Exception as e:
                st.error(f"Could not check this food.\n\n{e}")

st.divider()
st.caption(
    "Guidance only. For medical or dietary concerns about your pet, "
    "consult a qualified veterinarian."
)
