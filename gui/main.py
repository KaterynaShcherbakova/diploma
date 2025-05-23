import numpy as np
from ultralytics import YOLO
import gradio as gr
import cv2  

PRIMARY_MODEL_PATH = "yolo-v8-first.pt"   
SECONDARY_MODEL_PATH = "yolo-v11-second.pt"

model_primary = YOLO(PRIMARY_MODEL_PATH)
model_secondary = YOLO(SECONDARY_MODEL_PATH)

def classify_acne(image: np.ndarray) -> str:
    image = cv2.resize(image, (224, 224))
    results1 = model_primary.predict(image, task="classify", verbose=False)

    res1 = results1[0]

    probs1 = res1.probs.top1 
    names1 = model_primary.names

    top_class1 = names1[probs1]
    top_conf1 = res1.probs.top1conf

    lines = [
        f"Primary → {top_class1} ({top_conf1*100:.2f}%)"
    ]

    if top_class1.lower() != "acne":
        idx_acne = next(idx for idx, cls in names1.items() if cls.lower() == 'acne')
        lines.append(f"\nAcne detected with confidence only {res1.probs.data[idx_acne] * 100 :.2f}%")

    else:
        results2 = model_secondary.predict(image, task="classify", verbose=False)
        res2 = results2[0]

        probs2 = res2.probs 
        names2 = model_secondary.names

        lines.append("\nAcne-type probabilities:")
        for idx, name in names2.items():
            conf = float(probs2.data[idx] * 100)
            lines.append(f" • {name}: {conf:.2f}%")
            
    if top_class1.lower() != "clear":
            lines.append("\n\nWARNING!\nSelf-medication can be harmful to your health.\nPlease consider consulting a dermatologist.")

    return "\n".join(lines)

custom_primary = gr.themes.Color(
    c50="#FFDCDC",
    c100="#FEBDBD",
    c200="#FD7777",
    c300="#FD3535",
    c400="#ED0303",
    c500="#A80202", 
    c600="#880202",
    c700="#650101",
    c800="#420101",
    c900="#230000",
    c950="#0F0000"
)

theme = gr.themes.Soft(
    primary_hue=custom_primary,
    text_size=gr.themes.sizes.text_lg
)

iface = gr.Interface(
    fn=classify_acne,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Textbox(label="Results"),
    title="Acne Detector ➔ Type Classifier",
    description="1) Detects if an image contains acne.\n"
                "2) If acne detected, classifies its type with percentages.",
    theme=theme,
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()