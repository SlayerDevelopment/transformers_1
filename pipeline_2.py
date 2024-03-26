from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

if __name__ == "__main__":
    vision_classifier = pipeline(model="google/vit-base-patch16-224")
    preds = vision_classifier(
        images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    )
    for pred in preds:
        print({"score": round(pred["score"], 4), "label": pred["label"]})
