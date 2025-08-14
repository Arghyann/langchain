from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize the model and prompt
model = OllamaLLM(model="gemma3:1b")

template = '''You are an expert on answering questions about a Pizza restaurant.
Try to be as helpful as possible when answering the questions.

Here are some reviews from customers: {reviews}

Here is the question to answer: {question}
'''

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Interactive question answering loop
print("Restaurant Review Q&A System - Ask questions about our pizza restaurant!")
print("(Type 'q' to quit)")
print("-" * 60)

while True:
    question = input("\nEnter your question: ")
    if question.lower() == "q":
        print("\nThank you for using the Restaurant Review Q&A System!")
        break
    
    
    reviews = retriever.invoke(question)
    
    
    result = chain.invoke({"reviews": reviews, "question": question})
    print("\n" + "-" * 60)
    print("retrieved reviews: ", reviews)
    print("-" * 60)
    print("\n" + "-" * 60)
    print(result)
    print("-" * 60)
