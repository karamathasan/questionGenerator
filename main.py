import transformers
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from text import getNeurosciencePassage, getHistoryPassage

summarizer = pipeline('summarization')
model_name = "valhalla/t5-small-e2e-qg"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

'''
architecture:
    for now > T5,
    donators RAG,
    GPT,
    LangChain
procedure:
    suppose we have a large corpus of text that is the user's notes > 512 characters
    we need to process the text so that the question generator can handle the text better
    
    1)
        we can chunk the data into groups of 512 characters each, summarize them each and then generate a question on each summary
        - this may have trouble, because the questions might not be able to reflect the complexity and connections in the text
    
'''


def question_prompt(text, num_questions):
    return f"generate detailed question: {text}"
    # return f"I am a student. Generate question to access my memory: {text}"
    
def summarize_prompt(text):
    return f"summarize: {text}"


def getNumTokens(text):
    return len(text.split())
def chunk(text: str, chunk_size = 512):
    words = text.split()
    # Create chunks of words based on the word limit
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# def generate_summary(text):
#     summarized_inputs = summarize_prompt(text)
#     summarized_inputs = tokenizer.encode(summarized_inputs, return_tensors="pt")
#     summary_output = model.generate(summarized_inputs, max_new_tokens = 256)
#     summary_output = [tokenizer.decode(output, skip_special_tokens=True) for output in summary_output]
#     summary_output = (str(summary_output)).replace("<sep>", "")
#     return summary_output

def generate_summary(text):
    if getNumTokens(text) > 512:
        text = chunk(text)
    result = summarizer(text, max_length = 256, min_length = 102)
    return ' '.join([summary['summary_text'] for summary in result])

def generate_questions(text, num_questions):
    question_input = question_prompt(text, num_questions)
    question_input = tokenizer.encode(question_input, return_tensors="pt")
    question_outputs = model.generate(question_input, max_new_tokens = 256)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in question_outputs]
    questions = str(questions)
    return questions.split("<sep>")
    
def summarize_then_question(text,num_questions):
    summary = generate_summary(text)
    questions = generate_questions(summary,num_questions)
    return questions


print("\n\n")

summary = generate_summary(getHistoryPassage())
print(summary)

print(summarize_then_question(getHistoryPassage(), 10))

