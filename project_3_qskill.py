from groq import Groq
messages = []
while True:
 text = input("enter a text : ")
 if text == 'exit':
  print("okk byy")
  break

 client = Groq(api_key="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")
 chat = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{'role' : 'user', 'content' : text}]
    )
 print(chat.choices[0].message.content)

