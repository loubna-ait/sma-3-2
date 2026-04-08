from langchain.agents import create_agent

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool

load_dotenv(override=True)
chunks=[
    "Loubna Ait Hra, également connue sous le nom de Zubna Itera,"
    "est une passionnée d’informatique et de design graphique "
    "née à Jeddah et ayant grandi à Guelmim, au Maroc. Titulaire"
    "d’un baccalauréat en sciences physiques, d’un diplôme de "
    "technicien supérieur  et d’une licence en génie "
    "informatique de l’Université Ibn Zohr, elle a développé "
    "une expertise solide en programmation, bases de données "
    "et modélisation "
]

embedding_model = OpenAIEmbeddings() 
vector_store = Chroma.from_texts(
    texts=chunks,
    collection_name="cv_collection",
  #  embedding_function=embedding_model  # <- ici au lieu de embeddings
)
vector_store.embedding_function = embedding_model

retriever = vector_store.as_retriever()
retriever_tool=create_retriever_tool(
    retriever=retriever,
    name="cv_tool",
    description="get information about me "
    
)



llm = ChatOpenAI(model="gpt-4o", temperature=0)


@tool
def get_employee_info(name: str):
    """
    Get Infomration about emloyee (name, salary, seniority)
    """
  #  print("*" * 50)
    print("get_employee_info tool invoked")
   # print("*" * 50)
    return {"name": name, "salary": 12000, "seniority": 5}




# resp = agent.invoke(
#    input={"messages": [HumanMessage("Je veux connaitre le salaire de Yassine")]}
# )

# print(resp["messages"][-1].content)


def send_email(email:str,subject:str,content :str):
    """
    Send email with subject and content
    """
    print(f"Sending  email  to {email} , subject : {subject}, content :{content} ")
    return f"Email sent succefully to {email} , subject : {subject}, content :{content}"
llm=ChatOpenAI(model="gpt-4o",temperature=0)
agent = create_agent(
    model=llm,
    tools=[send_email ,get_employee_info,retriever_tool ],
    system_prompt="Answer to user query using provided tools"
)
resp = agent.invoke(
   input={"messages": [HumanMessage("quel est le salaire de yassine")]}
)
print(resp["messages"][-1].content)
