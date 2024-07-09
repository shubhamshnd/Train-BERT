import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

# Load the trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained("./qa_model")
tokenizer = BertTokenizerFast.from_pretrained("./qa_model")

# Function to get answer
def get_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def main():
    print("Welcome to the BERT QA chat! Type 'exit' to end the chat.")
    context = input("Provide the context for the questions: ")
    while True:
        question = input("Ask a question: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        answer = get_answer(question, context)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
