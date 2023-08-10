# Prediction interface for Cog ⚙️

from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM , AutoTokenizer

MODEL_NAME = "sahil2801/test3"
MODEL_CACHE = "model-cache/"
TOKEN_CACHE = "token-cache/"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKEN_CACHE,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True
        ).half().to("cuda")

    def predict(
        self,
        system_prompt: str = Input(description="System prompt", default="You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- \n{\n\t\"name\": \"plan_holiday\",\n\t\"description\": \"Plan a holiday based on user's interests\",\n\t\"parameters\": {\n\t\t\"type\": \"object\",\n\t\t\"properties\": {\n\t\t\t\"destination\": {\n\t\t\t\t\"type\": \"string\",\n\t\t\t\t\"description\": \"The destination of the holiday\"\n\t\t\t},\n\t\t\t\"duration\": {\n\t\t\t\t\"type\": \"integer\",\n\t\t\t\t\"description\": \"The duration of the trip in holiday\"\n\t\t\t}\n\t\t},\n\t\t\"required\": [\n\t\t\t\"destination\",\n\t\t\t\"duration\"\n\t\t]\n\t}\n}"),
        prompt : str = Input(description="User prompt", default="I am thinking of having a 10 day long vacation in Greece, can you help me plan it?"),
        new_tokens: int = Input(description="Generate at most this many new tokens in the response", ge=0, le=1024, default=128),
        temp: float = Input(description="Temperature to use", ge=0, le=1, default=0.5),
        top_p: float = Input(description="Temperature to use", ge=0, le=1, default=0.95),
    ) -> str:
        """Run a single prediction on the model"""
        full_prompt = f"SYSTEM: {system_prompt}. USER: {prompt}\n ASSISTANT:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=new_tokens
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output