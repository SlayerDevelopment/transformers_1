from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

if __name__ == "__main__":
    print("=" * 30, "START PIPELINES")
    transcriber_1 = pipeline(task="automatic-speech-recognition")
    transcriber_2 = pipeline(model="openai/whisper-large-v2")
    print("=" * 30, "START TRANSCRIBERS")
    aux_result_lst = []
    for t in transcriber_1(
        [
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        ]
    ):
        aux_result_lst.append(f'TASK:\t{t["text"]}')
    for t in transcriber_2(
        [
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        ]
    ):
        aux_result_lst.append(f'MODEL:\t{t["text"]}')
    print("=" * 30, "SHOW RESULTS 1")
    for r in aux_result_lst:
        print(r)
    print("=" * 30, "START PIPELINE WITH DATASET")
    pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")
    print("=" * 30, "SHOW RESULTS 2")
    for out in pipe(KeyDataset(dataset, "audio")):
        print(out)

