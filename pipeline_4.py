from transformers import pipeline

if __name__ == "__main__":
    classifier = pipeline(model="facebook/bart-large-mnli")
    classifier(
        "I have a problem with my iphone that needs to be resolved asap!!",
        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )
