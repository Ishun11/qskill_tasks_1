import speech_recognition as sr
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

recognizer = sr.Recognizer()

print("Loading Translation Model...")

translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="hin_Deva",
    tgt_lang="eng_Latn"
)

print("Loading Stable Diffusion Model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

print("Model Loaded Successfully!\n")


try:
    with sr.Microphone() as source:
        print("🎤 Speak something in Hindi...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)


    print("\nRecognizing Speech...")

    text = recognizer.recognize_google(
        audio,
        language="hi-IN"
    )

    print("\nRecognized Text:")
    print(text)

    print("\nTranslating...")

    translated = translator(text)

    prompt = translated[0]["translation_text"]

    print("\nEnglish Prompt:")
    print(prompt)

    print("\nGenerating Image... Please Wait...")

    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

    image.save("generated_image.png")

    print("\n Image Generated Successfully!")
    print("Saved as: generated_image.png")

except sr.UnknownValueError:
    print(" Could not understand the speech.")

except sr.RequestError as e:
    print(f" Google Speech Recognition Error:\n{e}")

except Exception as e:
    print(f" Error:\n{e}")