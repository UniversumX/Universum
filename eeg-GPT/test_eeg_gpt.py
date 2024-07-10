import os
from dotenv import load_dotenv
from eeg_gpt.agent import setup_eeg_gpt

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def main():
    eeg_gpt = setup_eeg_gpt()
    
    tasks = [
        "Analyze this EEG data and tell me if there are any abnormalities.",
        "What objects can you see in this image?",
        "Process this audio file and tell me what you hear.",
        "I have both EEG and image data. Can you analyze both and tell me if there's any correlation?",
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        result = eeg_gpt.run(task)
        print(f"Result: {result}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()