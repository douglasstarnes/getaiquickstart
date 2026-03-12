import tiktoken

text = """
Generative AI refers to a class of artificial intelligence systems designed to create new content—such as text, images, audio, video, or code—by learning patterns from large amounts of existing data. Unlike traditional AI, which focuses on classification or prediction, generative models produce original outputs that resemble human-created work. Technologies like large language models and diffusion models enable these systems to understand context, style, and structure, allowing them to generate coherent essays, realistic images, music, and even functional software code. This capability is driven by advances in deep learning, massive datasets, and scalable computing infrastructure.

The impact of generative AI is broad and rapidly expanding across industries. In software development, it accelerates coding, testing, and documentation; in creative fields, it supports ideation, design, and content production; and in business, it enhances customer support, data analysis, and decision-making. At the same time, generative AI raises important considerations around ethics, accuracy, intellectual property, and responsible use. As organizations adopt these tools, success increasingly depends on combining human judgment with AI-generated outputs to ensure quality, trust, and meaningful real-world value.
"""

enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode(text)

print(tokens)

decoded_text = enc.decode(tokens)
print(decoded_text)

if decoded_text == text:
  print("Original and decoded text are the same")
else:
  print("Something went wrong")

num_words = len(text.split(" "))
num_tokens = len(tokens)
print(f"The text has {num_words} and was encoded into {num_tokens} tokens.")
print(f"Ratio of words to tokens: {num_words / num_tokens}")
