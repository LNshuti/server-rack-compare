from langchain_community.llms import VLLM

llm = VLLM(model="mosaicml/mpt-7b",
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=128,
           top_k=10,
           top_p=0.95,
           temperature=0.8,
           # tensor_parallel_size=... # for distributed inference
)

print(llm("What is the capital of France ?"))

# from langchain_community.llms.fake import FakeStreamingListLLM
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import SystemMessagePromptTemplate
# from langchain_core.runnables import Runnable
# from operator import itemgetter

# prompt = (
#     SystemMessagePromptTemplate.from_template("You are a nice assistant.")
#     + "{question}"
# )
# llm = FakeStreamingListLLM(responses=["foo-lish"])

# chain: Runnable = prompt | llm | {"str": StrOutputParser()}

# chain_with_assign = chain.assign(hello=itemgetter("str") | llm)

# print(chain_with_assign.input_schema.schema())
# # {'title': 'PromptInput', 'type': 'object', 'properties':
# {'question': {'title': 'Question', 'type': 'string'}}}
# print(chain_with_assign.output_schema.schema()) #
# {'title': 'RunnableSequenceOutput', 'type': 'object', 'properties':
# {'str': {'title': 'Str',
# 'type': 'string'}, 'hello': {'title': 'Hello', 'type': 'string'}}}