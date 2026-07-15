
import os
import streamlit as st

# Read API key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from typing import TypedDict, List
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="Insurance Claims Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Insurance Claims Adjudication Agent")
st.write("Enter a claim and let the LangGraph agent analyze it.")

policies = [
    {"id":"policy_1","text":"Auto Policy A: Covers collision damage up to $10,000. Water damage is excluded unless caused by a covered collision. Deductible is $500."},
    {"id":"policy_2","text":"Auto Policy B: Covers water damage from flooding up to $5,000 if the policyholder has the Flood Endorsement add-on. Standard policy excludes flood damage entirely."},
    {"id":"policy_3","text":"State Regulation - California: Insurers must respond to claims within 15 business days. Denying a valid claim without justification can result in bad-faith penalties."},
]

@st.cache_resource
def build_retriever():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs=[Document(page_content=p["text"],metadata={"source":p["id"]}) for p in policies]
    chunks=splitter.split_documents(docs)
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore=Chroma.from_documents(chunks,embeddings,collection_name="insurance_policies")
    return vectorstore.as_retriever(search_kwargs={"k":3})

retriever=build_retriever()
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

class ClaimState(TypedDict,total=False):
    claim:str
    query:str
    retrieved_docs:List[str]
    relevance_score:str
    retry_count:int
    decision:str
    reasoning:str
    grounded:bool

def retrieve_node(state):
    q=state.get("query") or state["claim"]
    state["retrieved_docs"]=[d.page_content for d in retriever.invoke(q)]
    return state

def grade_node(state):
    docs="\n".join(state["retrieved_docs"])
    r=llm.invoke(f'Claim:{state["claim"]}\nPolicy:{docs}\nDoes policy contain enough info? Answer yes or no.')
    state["relevance_score"]=r.content.strip().lower()
    return state

def rewrite_node(state):
    r=llm.invoke(f'Rewrite search query for claim: {state["claim"]}')
    state["query"]=r.content.strip()
    state["retry_count"]=state.get("retry_count",0)+1
    return state

def decide_node(state):
    docs="\n".join(state["retrieved_docs"])
    r=llm.invoke(f'Claim:{state["claim"]}\nPolicy:{docs}\nDecide approve, deny or escalate. First line decision, second line reason.')
    parts=r.content.strip().split("\n",1)
    state["decision"]=parts[0].lower()
    state["reasoning"]=parts[1] if len(parts)>1 else ""
    return state

def grounding_node(state):
    docs="\n".join(state["retrieved_docs"])
    r=llm.invoke(f'Reasoning:{state["reasoning"]}\nPolicy:{docs}\nGrounded? yes or no')
    state["grounded"]="yes" in r.content.lower()
    return state

def escalate_node(state):
    state["decision"]="escalate"
    state["reasoning"]="Insufficient or ungrounded evidence. Escalated to a human adjuster."
    return state

MAX_RETRIES=2
def relevance_router(state):
    if "yes" in state["relevance_score"]:
        return "decide"
    if state.get("retry_count",0)>=MAX_RETRIES:
        return "escalate"
    return "rewrite"

def grounding_router(state):
    return "end" if state["grounded"] else "escalate"

graph=StateGraph(ClaimState)
graph.add_node("retrieve",retrieve_node)
graph.add_node("grade",grade_node)
graph.add_node("rewrite",rewrite_node)
graph.add_node("decide",decide_node)
graph.add_node("ground",grounding_node)
graph.add_node("escalate",escalate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve","grade")
graph.add_conditional_edges("grade",relevance_router,{
    "decide":"decide","rewrite":"rewrite","escalate":"escalate"})
graph.add_edge("rewrite","retrieve")
graph.add_edge("decide","ground")
graph.add_conditional_edges("ground",grounding_router,{
    "end":END,"escalate":"escalate"})
graph.add_edge("escalate",END)
agent=graph.compile()

claim=st.text_area("Enter insurance claim",height=150)

if st.button("Analyze Claim"):
    if claim.strip():
        with st.spinner("Analyzing..."):
            result=agent.invoke({"claim":claim,"retry_count":0})
        st.subheader("Decision")
        st.success(result.get("decision",""))
        st.subheader("Reasoning")
        st.write(result.get("reasoning",""))
        st.subheader("Retrieved Policy")
        for d in result.get("retrieved_docs",[]):
            st.code(d)
    else:
        st.warning("Please enter a claim.")
