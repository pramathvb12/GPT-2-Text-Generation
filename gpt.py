from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_name='gpt2', max_length=50):
    # Load pre-trained GPT model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text based on the input prompt
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Example usage
prompt = input("Enter the text")
generated_text = generate_text(prompt, model_name='gpt2', max_length=100)
print(generated_text)
