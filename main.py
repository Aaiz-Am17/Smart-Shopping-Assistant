# main.py
import customtkinter as ctk
from gui_app import ProductChoiceInterface
from ac_model import ACModel
from tv_model import TVModel
from google.generativeai import Client as GenerativeAIClient

# Dataset paths (UPDATE THESE TO YOUR ACTUAL PATHS)
AC_FILE_PATH = "Dataset/Air_condition_dataset.csv"
TV_FILE_PATH = "Dataset/TELEVISION.csv"

# Google Generative AI API Key
API_KEY = "Add Your Own API KEY" 
def main():
    # Initialize Google Generative AI Client
    try:
        ai_client = GenerativeAIClient(api_key=API_KEY)
    except Exception as e:
        print(f"Error initializing Google Generative AI client: {e}")
        # Handle this error (e.g., exit, or run without AI features)
        ai_client = None

    # Initialize models
    ac_model = ACModel(AC_FILE_PATH)
    tv_model = TVModel(TV_FILE_PATH)

    root = ctk.CTk()
    app = ProductChoiceInterface(root, ac_model, tv_model, ai_client)
    root.mainloop()

if __name__ == "__main__":
    main()
