
from dpreprocess import dataset 

# Define a function to answer questions using the fine-tuned model
def answer_question(question):
    inputs = tokenizer.encode_plus(
        question,
        preprocessed_docs,
        add_special_tokens=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate predictions
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the start and end positions of the answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Get the answer span
    answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index+1])
    return answer

# Example usage
question = "What is the experience of the candidate?"
answer = answer_question(question)
print("Answer:", answer)
