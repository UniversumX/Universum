from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain_core.prompts import BaseChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from typing import List, Union
import re
from .tools import EEGProcessingTool, VisualRecognitionTool, AudioProcessingTool

from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from .tools import EEGProcessingTool, VisualRecognitionTool, AudioProcessingTool


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        action_match = re.search(r"Action: (.*?)[\n]*Thought:", llm_output, re.DOTALL)
        if not action_match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = action_match.group(1).strip()
        
        thought_match = re.search(r"Thought: (.*?)(?:\n|$)", llm_output, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        return AgentAction(tool=action, tool_input=thought, log=llm_output)


def setup_eeg_gpt():
    llm = ChatOpenAI(temperature=0)
    
    tools = [
        Tool(
            name="EEG Processing",
            func=EEGProcessingTool().run,
            description="Useful for processing and analyzing EEG data"
        ),
        Tool(
            name="Visual Recognition",
            func=VisualRecognitionTool().run,
            description="Useful for recognizing objects or scenes in images"
        ),
        Tool(
            name="Audio Processing",
            func=AudioProcessingTool().run,
            description="Useful for processing and analyzing audio data"
        )
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms!"
    )

    return agent

