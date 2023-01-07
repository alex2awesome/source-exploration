# # Load a trained model and vocabulary that you have fine-tuned
# model = GPT2LMHeadModel.from_pretrained(output_dir)
# tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
# model.to(device)

"""# Generate Text"""

model.eval()

prompt = "<|startoftext|>"

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)

print(generated)

sample_outputs = model.generate(
    generated,
    # bos_token_id=random.randint(1,30000),
    do_sample=True,
    top_k=50,
    max_length=300,
    top_p=0.95,
    num_return_sequences=3
)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
