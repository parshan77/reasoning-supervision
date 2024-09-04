from openai import OpenAI
client = OpenAI()

# finetune_filename = "finetune_37.jsonl"
# uploaded_filename = "file-zTTSS4zWZf6JH6lPmDpsh82g"

# finetune_filename = "finetune_37_completion.jsonl"
# uploaded_filename = "file-UbKEHCgJyalCXB2ialgPoTsY"

# finetune_filename = "finetune_40_chat.jsonl"
# uploaded_filename = "file-gZry7DflgaGtfoQzocAhDALZ"

# finetune_filename = "0125_correct.jsonl"
# uploaded_filename = "file-dnMLuIPWGdNG0LybjFBv41Ur"

# finetune_filename = "0125_correct_fix.jsonl"
# uploaded_filename = "file-iDqd1Y3IK2Z4aVJTxeEAGB5g"

# finetune_filename = "0125_correct_ex.jsonl"
# uploaded_filename = "file-jBHY2q0JX0qdxPYlhnwexwoY"

filename = "file-FB7YdMLIV0HLlU8x8ADw4Lsu"

if __name__ == "__main__":
    # create finetune job
    client.fine_tuning.jobs.create(
        training_file=filename, 
        model="gpt-3.5-turbo-0125"
    )