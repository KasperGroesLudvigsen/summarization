from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


MODEL_ID = "google/gemma-2-27b-it"

model = SparseAutoModelForCausalLM.from_pretrained(
  MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)