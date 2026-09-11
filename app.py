import json
import os
import random
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path(*parts):
    return os.path.join(BASE, *parts)


with open(path("classes.json")) as f:
    pet_classes = json.load(f)
with open(path("fish_classes.json")) as f:
    fish_classes = json.load(f)
with open(path("bird_classes.json")) as f:
    bird_classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(weights, num_classes):
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    state = torch.load(path(weights), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


pet_model = load_model("final_pet_model.pth", len(pet_classes))
fish_model = load_model("fish_model.pth", len(fish_classes))
bird_model = load_model("bird_model.pth", len(bird_classes))

HEADS = {
    "Dog/Cat": (pet_model, pet_classes),
    "Fish": (fish_model, fish_classes),
    "Bird": (bird_model, bird_classes),
}


def predict_breed(image, animal_type):
    if image is None:
        return "Please upload an image first."
    model, classes = HEADS[animal_type]
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    confidence, index = torch.max(probs, 1)
    name = classes[index.item()].replace("_", " ")
    if name == "Tzu":
        name = "Shih Tzu"
    conf = confidence.item() * 100
    note = "" if conf >= 60 else "\n\nLow confidence — try a clearer, closer photo."
    return f"Breed / species: {name}\nConfidence: {conf:.2f}%{note}"


from transformers import BertForSequenceClassification, BertTokenizer  # noqa: E402

food_df = pd.read_csv(path("pet_food_dataset.csv"))
food_tokenizer = BertTokenizer.from_pretrained(path("food_bert_model"))
food_model = BertForSequenceClassification.from_pretrained(path("food_bert_model"))
food_model.eval()

FOOD_LABELS = ["Safe", "Moderate", "Unsafe"]


def check_food(pet_type, pet_size, food):
    if not food or not food.strip():
        return "Please enter some ingredients."
    food = food.lower()
    subset = food_df[
        (food_df["pet_type"].str.lower() == pet_type.lower())
        & (food_df["pet_category"].str.lower() == pet_size.lower())
    ]
    for _, row in subset.iterrows():
        if str(row["food_name"]).lower() in food:
            label = str(row["label"]).strip().lower()
            if label == "safe":
                return ("SAFE\n\nThis food is suitable for your pet.\n\n"
                        "Other good options:\n- Chicken\n- Salmon\n- Pumpkin")
            if label == "moderate":
                return ("MODERATE\n\nGive this only occasionally.\n"
                        "Large or frequent servings may cause problems.")
            safe = subset[subset["label"].str.strip().str.lower() == "safe"]["food_name"].tolist()
            if len(safe) < 4:
                safe = food_df[food_df["label"].str.strip().str.lower() == "safe"]["food_name"].sample(4).tolist()
            bullets = "\n".join(f"- {s}" for s in safe[:4])
            return f"UNSAFE\n\nThis food may harm your pet.\n\nSafer choices:\n{bullets}"
    text = f"{pet_type} {pet_size} {food}"
    inputs = food_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = food_model(**inputs).logits
    pred = torch.argmax(logits).item()
    return f"{FOOD_LABELS[pred].upper()}  (model prediction)\n\nNot in the reference table; classified by the model."


from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

products = pd.read_csv(path("pet_products.csv"))
products["text"] = (products["product_name"] + " "
                    + products["pet_type"] + " "
                    + products["pet_category"])
encoder = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = encoder.encode(products["text"].tolist())


def recommend_products(pet_type, breed, size, age, weight):
    if size == "big":
        size = "large"
    query = f"{pet_type} {breed} {size} pet products"
    scores = cosine_similarity(encoder.encode([query]), product_embeddings)[0]
    top = scores.argsort()[-6:][::-1]
    gallery = []
    for idx in top:
        row = products.iloc[idx]
        candidates = [row[c] for c in ("image1", "image2", "image3")
                      if c in products.columns and isinstance(row[c], str)]
        if not candidates:
            continue
        filename = os.path.basename(random.choice(candidates))
        img = path("productimages", filename)
        if os.path.exists(img):
            gallery.append((img, row["product_name"]))
    return gallery


# --------------------------------------------------------------------------
# Larger type. Layout is unchanged.
# --------------------------------------------------------------------------

CUSTOM_CSS = """
/* Modest bump only. Default Gradio theme, spacing and styling are untouched. */

#app-title h1      { font-size: 1.95rem !important; }
#app-subtitle p    { font-size: 1.05rem !important; }

button[role="tab"] { font-size: 1.05rem !important; }

label span,
span[data-testid="block-info"] { font-size: 1rem !important; }

input[type="text"],
input[type="number"],
textarea,
select             { font-size: 1rem !important; }

#breed-output textarea,
#food-output textarea { font-size: 1.05rem !important; line-height: 1.55 !important; }

.gr-button, button.primary { font-size: 1.05rem !important; }

#app-footer p      { font-size: 0.95rem !important; }
"""

with gr.Blocks(
    title="Pet Breed Identification and Product Recommender",
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown("# Pet Breed Identification and Product Recommender", elem_id="app-title")
    gr.Markdown(
        "Identify your pet, check whether a food is suitable, and get product suggestions.",
        elem_id="app-subtitle",
    )

    with gr.Tab("Breed Identification"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Upload a pet photo")
                animal_type = gr.Radio(["Dog/Cat", "Fish", "Bird"],
                                       label="Animal type", value="Dog/Cat")
                identify_btn = gr.Button("Identify", variant="primary")
            with gr.Column():
                breed_out = gr.Textbox(label="Prediction", lines=4, elem_id="breed-output")
        identify_btn.click(predict_breed, [image_in, animal_type], breed_out)

    with gr.Tab("Product Recommendation"):
        with gr.Row():
            with gr.Column():
                p_type = gr.Dropdown(["dog", "cat", "bird", "fish"], label="Pet type", value="dog")
                p_breed = gr.Textbox(label="Breed / species")
                p_size = gr.Dropdown(["small", "big", "kitten", "adult"], label="Size / life stage", value="adult")
                p_age = gr.Number(label="Age (years)", value=1)
                p_weight = gr.Number(label="Weight (kg)", value=5)
                rec_btn = gr.Button("Recommend", variant="primary")
            with gr.Column():
                gallery = gr.Gallery(label="Recommended products", columns=2, height=420)
        rec_btn.click(recommend_products, [p_type, p_breed, p_size, p_age, p_weight], gallery)

    with gr.Tab("Food Quality Check"):
        with gr.Row():
            with gr.Column():
                f_type = gr.Dropdown(["dog", "cat", "bird", "fish"], label="Pet type", value="dog")
                f_size = gr.Dropdown(["small", "big", "kitten", "adult"], label="Size / life stage", value="adult")
                f_text = gr.Textbox(label="Food ingredients",
                                    placeholder="e.g. chicken meal, brown rice")
                food_btn = gr.Button("Check", variant="primary")
            with gr.Column():
                food_out = gr.Textbox(label="Result", lines=10, elem_id="food-output")
        food_btn.click(check_food, [f_type, f_size, f_text], food_out)

    gr.Markdown(
        "_Guidance only. For medical or dietary concerns about your pet, consult a veterinarian._",
        elem_id="app-footer",
    )

if __name__ == "__main__":
    demo.launch()
