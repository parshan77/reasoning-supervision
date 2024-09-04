from openai import OpenAI
client = OpenAI()

filename = "0125_correct_ex_fix.jsonl"

if __name__ == "__main__":
    # upload file
    client.files.create(
        file=open(filename, "rb"),
        purpose="fine-tune"
    )