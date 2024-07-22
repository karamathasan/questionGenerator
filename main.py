import transformers
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from text import getNeurosciencePassage

model_name = "valhalla/t5-small-e2e-qg"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def question_prompt(text, num_questions):
    return f"generate detailed question: {text}"
    # return f"I am a student. Generate question to access my memory: {text}"
    
def summarize_prompt(text):
    return f"summarize: {text}"

# def chunk

def generate_summary(text):
    summarized_inputs = summarize_prompt(text)
    summarized_inputs = tokenizer.encode(summarized_inputs, return_tensors="pt")
    summary_output = model.generate(summarized_inputs, max_new_tokens = 256)
    summary_output = [tokenizer.decode(output, skip_special_tokens=True) for output in summary_output]
    summary_output = (str(summary_output)).replace("<sep>", "")
    return summary_output

def generate_questions(text, num_questions):
    question_input = question_prompt(text, num_questions)
    question_input = tokenizer.encode(question_input, return_tensors="pt")
    question_outputs = model.generate(question_input, max_new_tokens = 256)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in question_outputs]
    questions = str(questions)
    return questions.split("<sep>")
    
def summarize_question(text,num_questions):
    summary = generate_summary(text)
    questions = generate_questions(summary,num_questions)
    return questions

text = getNeurosciencePassage()
questions = generate_questions(text,5)
for question in questions:
    print(question)
print("\n\n")

questions = summarize_question(text,5)
for question in questions:
    print(question)