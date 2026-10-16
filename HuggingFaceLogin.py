from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

login(token=os.getenv("HUGGING_FACE_TOKEN"))