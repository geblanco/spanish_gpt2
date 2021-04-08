from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

data_dir = "models/gpt2_es"
tokenizer = AutoTokenizer.from_pretrained(data_dir)
model = AutoModelWithLMHead.from_pretrained(data_dir)
tokenizer.model_max_length=1024
model.eval()

text = "Quién era Jim Henson? Jin Henson era un"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss, logits = outputs[:2]
predicted_index = torch.argmax(logits[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_index])
print('input text:', text)
print('predicted text:', predicted_text)
sample_outputs = model.generate(inputs.input_ids,
                                pad_token_id=50256,
                                do_sample=True,
                                max_length=50, # put the token number you want
                                top_k=40,
                                num_return_sequences=1)

for i, sample_output in enumerate(sample_outputs):
    print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist())))


