## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
LangChain Expression Language (LCEL) simplifies interactions with large language models (LLMs) by creating reusable and structured expressions. This task involves:

1. Designing an LCEL expression with dynamic prompt parameters (e.g., topic and length).
2. Using three essential components: Prompt- A structured input with placeholders for parameters, Model- An LLM used to process the prompt and Output Parser- A parser to interpret the model's output.
3. Demonstrating the LCEL expression's functionality in generating structured, relevant outputs.

### DESIGN STEPS:

#### STEP 1: Define the Parameters
Identify the parameters (topic and length) to allow dynamic customization of prompts.

#### STEP 2: Design the Prompt Template
Create a structured prompt template with placeholders for parameters.

#### STEP 3: Select the Model
Use an LLM, such as OpenAI's GPT, to process the prompt.

#### STEP 4: Implement the Output Parser
Design an output parser to format and structure the model's output.

#### STEP 5: Integrate Components into an LCEL Expression
Combine the prompt template, model, and output parser into a LangChain pipeline.

#### STEP 6: Evaluate with Examples
Test the LCEL expression using multiple input values for topic and length.

### PROGRAM:
```
Name:Prathik TS
Reg.No:21222224240117
```
```
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from google.colab import userdata
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Step 1: Define Parameters and Prompt Template
prompt_template = PromptTemplate(
    input_variables=["topic", "length"],
    template=(
        "You are a helpful assistant. Please provide the following in valid JSON format:\n"
        "- A concise summary of {topic}.\n"
        "- The word count of the summary.\n\n"
        "Response should look like this:\n"
        "{{\n"
        '  "summary": "Your concise summary here.",\n'
        '  "word_count": 123\n'
        "}}\n\n"
        "Write a {length}-word summary about {topic}. Be concise and factual."
    )
)
```
```
# Step 2: Define the Output Parser
response_schemas = [
    ResponseSchema(name="summary", description="A concise summary of the topic."),
    ResponseSchema(name="word_count", description="The number of words in the summary."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Step 3: Create the LangChain LLM Chain with Gemini Model
API_KEY = userdata.get('GEMINI_API_KEY') 
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    temperature=0.3, 
    google_api_key=API_KEY
)

chain = LLMChain(prompt=prompt_template, llm=llm, output_parser=output_parser)

# Step 4: Execute the Chain with Examples
examples = [
    {"topic": "Climate Change", "length": "50"},
    {"topic": "Artificial Intelligence", "length": "30"},
]

for example in examples:
    try:
        result = chain.run(example)
        print(f"Input: {example}")
        print(f"Output: {result}\n")
    except Exception as e:
        print(f"Error for input {example}: {e}\n")

```
### OUTPUT:
<img width="1869" height="187" alt="390028985-05163bf9-5013-473f-8a64-21a68b1712b3" src="https://github.com/user-attachments/assets/d8bcd928-8595-406a-93b4-43d806b49d6a" />

### RESULT:
  Thus, the LangChain Expression Language (LCEL) expression that utilizes two prompt parameters and three key components (prompt, model, and output parser) was designed and implemented successfully. And also evaluated its functionality by analyzing relevant examples of its application in real-world scenarios.
