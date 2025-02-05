import os
from glob import glob
from typing import List
from transformers.agents.default_tools import Tool
from transformers.agents import Toolbox
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from da_agent.agent.generatedtool import parse_generated_tools

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
EMBED_MODEL_TYPE = os.getenv("EMBED_MODEL_TYPE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ToolRetriever:
    def __init__(self, generated_tool_dir: str):
        self.generated_tool_dir = generated_tool_dir
        self.vectordb_path = os.path.join(self.generated_tool_dir, "vectordb")
        self.similarity_threshold = 0.6  # 添加相似度阈值

        if not os.path.exists(self.vectordb_path):
            os.makedirs(self.vectordb_path)

        # 初始化向量数据库和嵌入函数
        # 根据您的实际情况，设置 embedding_function
     
        if EMBED_MODEL_TYPE == "OpenAI":
            embedding_function = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY, #x修改后的
                model="text-embedding-3-large",
                base_url="https://api2.aigcbest.top/v1",
            )
            embed_model_name = EMBED_MODEL_NAME

        self.vectordb = Chroma(
            collection_name="tool_vectordb",
            embedding_function=embedding_function,
            persist_directory=self.vectordb_path,
        )

        self.generated_tools = {}
        for path in glob(os.path.join(generated_tool_dir, "*.py")):
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            tools = parse_generated_tools(code)
            for tool in tools:
                self.generated_tools[tool.name] = tool

    # def retrieve(self, query: str, k: int = 1) -> List[Tool]:
    #     k = min(len(self.vectordb), k)
    #     if k == 0:
    #         print("No tools in the database.")
    #         return []
    #     docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
    #     tools = []
    #     for doc, _ in docs_and_scores:
    #         name = doc.metadata["name"]
    #         if not name:
    #             print("Tool name not found in document metadata.")
    #             continue
    #         if name not in self.generated_tools:
    #             print(f"Tool '{name}' not found in generated_tools.")
    #             continue
    #         tools.append(self.generated_tools[name])
    #     return tools
    

    def retrieve(self, query: str, k: int = 5) -> List[Tool]:
        # 限制 k 的值为 5 或数据库中工具数量的最小值
        k = min(len(self.vectordb), k)
        if k == 0:
            print("No tools in the database.")
            return []

        # 执行相似度检索，返回文档及其分数
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        lowest_score_tool = None
        lowest_score = float('inf')

        for doc, score in docs_and_scores:
            name = doc.metadata["name"]
            if not name:
                # print("Tool name not found in document metadata.")
                continue
            if name not in self.generated_tools:
                # print(f"Tool '{name}' not found in generated_tools.")
                continue

            # 更新最低分工具
            if score < lowest_score:
                lowest_score = score
                lowest_score_tool = self.generated_tools[name]

        # 返回包含最低分工具的列表
        if lowest_score_tool:
            # print(f"Lowest score tool: {lowest_score_tool.name} with score {lowest_score}")
            return [lowest_score_tool]
        else:
            print("No valid tools found.")
            return []


    def add_new_tool(self, tool: Tool):
        program_name = tool.name
        program_description = tool.description
        # program_code = tool.code
        program_inputs = tool.inputs
        program_output_type = tool.output_type
        # program_dependencies = tool.dependencies

        res = self.vectordb._collection.get(ids=[program_name])
        # 检查工具是否已存在
        if res["ids"]:
            # raise ValueError(f"Tool {program_name} already exists!")
            print(f"Tool {program_name} already exists, deleting the old one...")
            # 如果需要覆盖，可以删除并重新添加
            self.vectordb._collection.delete(ids=[program_name])

        # 添加新工具到向量数据库和工具字典
        self.vectordb.add_texts(
            texts=[program_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )
        self.generated_tools[tool.name] = tool

        # self.vectordb.persist()

    def add_new_tool_from_path(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        tools = parse_generated_tools(code)
        for tool in tools:
            self.add_new_tool(tool)

    def retrieve_des(self, query: str, k: int = 1) -> List[Tool]:
        k = min(len(self.vectordb), k)
        if k == 0:
            print("No tools in the database.")
            return []
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        results = []
        for doc,score in docs_and_scores:
            name = doc.metadata["name"]
            if name in self.generated_tools:
                results.append((self.generated_tools[name], score))
        return results
           


    def get_tool_code(self, tool_name: str):
        """
        根据工具名称检索其代码。
        
        Args:
            tool_name (str): 工具的名称。
        
        Returns:
            Optional[str]: 工具的代码，如果未找到则返回 None。
        """
        # 假设生成的工具存储在 generated_tool_dir 目录下，文件名格式为 "{id}_{tool_name}.py"
        pattern = os.path.join(self.generated_tool_dir, f"*_{tool_name}.py")
        matched_files = glob(pattern)
        if not matched_files:
            print(f"Tool code file for '{tool_name}' not found.")
            return None
        file_path = matched_files[0]  # 假设每个工具只有一个代码文件
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return code
        except Exception as e:
            print(f"Error reading tool code for '{tool_name}': {str(e)}")
            return None


class ToolRetrievalTool(Tool):
    name = "get_relevant_tools"
    description = 'This tool retrieves relevant tools generated in previous runs. Provide a query describing what you want to do. If there are no tools in the toolbox, "No tool found" will be returned.'
    inputs = "query: str"
    output_type = "str"

    def __init__(self, generated_tool_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_tool_dir = generated_tool_dir
        self.tool_retriever = ToolRetriever(generated_tool_dir)
        self.tool_description_template = "<<tool_name>>: <<tool_description>>"

    def forward(self, query: str) -> str:
        relevant_tools: List[Tool] = self.tool_retriever.retrieve(query)
        if relevant_tools:
            relevant_toolbox = Toolbox(relevant_tools)
            descriptions = relevant_toolbox.show_tool_descriptions(
                self.tool_description_template
            )
            # 确保占位符被替换
            if "<<tool_name>>" in descriptions or "<<tool_description>>" in descriptions:
                descriptions = "\n".join(
                    f"{tool.name}: {tool.description}" for tool in relevant_tools
                )
            return descriptions
            # return relevant_toolbox.show_tool_descriptions(
            #     self.tool_description_template
            # )
        else:
            return "No tool found"

    def add_new_tool_from_path(self, path: str):
        return self.tool_retriever.add_new_tool_from_path(path)