import json
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_prompt

# Load environment variables
load_dotenv()

# Constants
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

# Initialize Pinecone vectorstore
vectorstore = Pinecone.from_existing_index(
    PINECONE_INDEX_NAME,
    OpenAIEmbeddings(
        api_key=os.getenv('OPENAI_EMBEDDING_API_KEY'),
        model=os.getenv('OPENAI_EMBEDDING_MODEL'),
        base_url=os.getenv('OPENAI_EMBEDDING_BASE_URL')
    )
)

def ask_question(question: str, filter: dict[str, str] = {}, max_docs: int = None) -> str:
    """Ask a question with the specified filters and document constraints."""
    return ask_question_with_prompt_file('question.prompt.txt', question, filter, max_docs)

def get_quiz(title: str, constraint: str = '') -> dict[str, str | list[str]]:
    """Generate a quiz for the given title with constraints."""
    print(f'Fetching quiz for: {title}')
    filter = {'title': title}
    answer = ask_question_with_prompt_file('get_quiz.prompt.txt', constraint, filter)

    try:
        quiz = parse_quiz_answer(answer)
        quiz['title'] = title
        print(f'Quiz: {quiz}')
        return quiz
    except Exception as e:
        print(f'Error parsing quiz: {e}')
        raise Exception('Error generating quiz. Please try again.')  # For end-user clarity

def parse_quiz_answer(answer: str) -> dict[str, str | list[str]]:
    """Parse the quiz answer from raw response."""
    answer = answer.lstrip('```json').rstrip('```')
    quiz_data = json.loads(answer).get('quiz', {})
    if not is_valid_quiz(quiz_data):
        raise ValueError("Invalid quiz data.")
    return quiz_data

def is_valid_quiz(quiz_data: dict) -> bool:
    """Check if the quiz data contains valid fields."""
    return (
        quiz_data.get('question') and
        quiz_data.get('choices') and len(quiz_data['choices']) >= 3 and
        quiz_data.get('answer') and quiz_data['answer'] in quiz_data['choices']
    )

def ask_question_with_prompt_file(prompt_file: str, question: str, filter: dict[str, str], max_docs: int) -> str:
    """Fetch answer to a question based on a prompt and data filters."""
    search_kwargs = {'filter': filter} if filter else {}
    if max_docs:
        search_kwargs['k'] = max_docs  # Default to 4 if not specified
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs=search_kwargs)

    # Define the RAG (Retrieval-Augmented Generation) prompt
    template = load_prompt(prompt_file)
    prompt = ChatPromptTemplate.from_template(template)

    # Create the RAG model
    model = initialize_model()

    # Build and execute RAG chain
    chain = create_rag_chain(retriever, prompt, model)
    print(f"Executing RAG with Question: {question}")
    return execute_rag_chain(chain, question)

def initialize_model() -> ChatOpenAI:
    """Initialize and return the ChatOpenAI model with configuration."""
    return ChatOpenAI(
        temperature=0,
        api_key=os.getenv('OPENAI_API_KEY2'),
        model=os.getenv('OPENAI_MODEL2'),
        base_url=os.getenv('OPENAI_BASE_URL2'),
        max_tokens=int(os.getenv('MAX_TOKENS2'))
    )

def create_rag_chain(retriever, prompt, model) -> RunnableParallel:
    """Create the RAG chain to process the question."""
    return (
        RunnableParallel({'context': retriever, 'question': RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    ).with_types(input_type=str)

def execute_rag_chain(chain, question: str) -> str:
    """Invoke the RAG chain to get the answer."""
    try:
        return chain.invoke(question)
    except Exception as e:
        print(f"Error with primary model: {e}. Trying fallback model...")
        return retry_with_fallback_model(chain, question)

def retry_with_fallback_model(chain, question: str) -> str:
    """Attempt to execute the question with a fallback model if the primary fails."""
    fallback_model = initialize_fallback_model()
    fallback_chain = create_rag_chain(chain.context, chain.prompt, fallback_model)
    try:
        return fallback_chain.invoke(question)
    except Exception as e:
        print(f"Error with fallback model: {e}")
        return "Sorry, I can't find the answer. Please try asking another question."

def initialize_fallback_model() -> ChatOpenAI:
    """Initialize and return the fallback ChatOpenAI model."""
    return ChatOpenAI(
        temperature=0,
        api_key=os.getenv('OPENAI_API_KEY'),
        model=os.getenv('OPENAI_MODEL'),
        base_url=os.getenv('OPENAI_BASE_URL'),
        max_tokens=int(os.getenv('MAX_TOKENS'))
    )

if __name__ == '__main__':
    # Example test case for quizzes
    titles = [
        'Developing Clinical Risk Prediction Models for Worsening Heart Failure Events and Death by Left Ventricular Ejection Fraction',
        'Burden of Illness beyond Mortality and Heart Failure Hospitalizations in Patients Newly Diagnosed with Heart Failure in Spain According to Ejection Fraction',
        'Prevalence, Characteristics, Management and Outcomes of Patients with Heart Failure with Preserved, Mildly Reduced, and Reduced Ejection Fraction in Spain',
        '20 Years of Real-World Data to Estimate the Prevalence of Heart Failure and Its Subtypes in an Unselected Population of Integrated Care Units'
    ]

    for title in titles:
        quiz = get_quiz(title)

    # Example test case for questions
    questions = [
        'Summarize research findings on heart failure.',
        'What are current research findings on heart failure?',
        'What are the latest treatments for COVID-19?',
        'What treatments are there for osteoporosis?',
        'Describe in detail the research in osteoporosis treatments.'
    ]

    for question in questions:
        answer = ask_question(question)
        print(f"Answer: {answer}")
