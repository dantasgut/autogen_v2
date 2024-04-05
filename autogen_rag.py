import os
from typing_extensions import Annotated
import chromadb

import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


#config_list = autogen.config_list_from_json("CONFIG_LIST")
config_list = [
    {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]},
]

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="The boss who ask questions and give tasks.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
    description="Assistant who has extra content retrieval power for solving difficult problems.",
)

coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer, you provide python code to answer questions. Reply `TERMINATE` in the end when everything is done.",
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
    description="Senior Python Engineer who can write code to solve problems and answer questions.",
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
    description="Product Manager who can design and plan the project.",
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
    description="Code Reviewer who can review the code.",
)

PROBLEM = "How to use spark for parallel training in FLAML? Give me sample code."


def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss_aid, pm, coder, reviewer], messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config={"config_list": config_list, "timeout": 60, "temperature": 0}
    )

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        message=boss_aid.message_generator,
        problem=PROBLEM,
        n_results=3,
    )

rag_chat()

