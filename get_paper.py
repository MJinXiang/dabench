# import openreview
# import re
# import csv
# from typing import Union, List, Dict
# from datetime import datetime


# def get_submissions(client, venue_id, status='all'):
#     # Retrieve the venue group information
#     venue_group = client.get_group(venue_id)
    
#     # Define the mapping of status to the respective content field
#     status_mapping = {
#         "all": venue_group.content['submission_name']['value'],
#         "accepted": venue_group.id,  # Assuming 'accepted' status doesn't have a direct field
#         "under_review": venue_group.content['submission_venue_id']['value'],
#         "withdrawn": venue_group.content['withdrawn_venue_id']['value'],
#         "desk_rejected": venue_group.content['desk_rejected_venue_id']['value']
#     }

#     # Fetch the corresponding submission invitation or venue ID
#     if status in status_mapping:
#         if status == "all":
#             # Return all submissions regardless of their status
#             return client.get_all_notes(invitation=f'{venue_id}/-/{status_mapping[status]}')
        
#         # For all other statuses, use the content field 'venueid'
#         return client.get_all_notes(content={'venueid': status_mapping[status]})
    
#     raise ValueError(f"Invalid status: {status}. Valid options are: {list(status_mapping.keys())}")




# def extract_submission_info(submission):
#     # Helper function to convert timestamps to datetime
#     def convert_timestamp_to_date(timestamp):
#         return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d') if timestamp else None

#     # Extract the required information
#     submission_info = {
#         'id': submission.id,
#         'title': submission.content['title']['value'],
#         'abstract': submission.content['abstract']['value'],
#         'keywords': submission.content['keywords']['value'],
#         'primary_area': submission.content['primary_area']['value'],
#         'TLDR': submission.content['TLDR']['value'] if 'TLDR' in submission.content else "",
#         'creation_date': convert_timestamp_to_date(submission.cdate),
#         'original_date': convert_timestamp_to_date(submission.odate),
#         'modification_date': convert_timestamp_to_date(submission.mdate),
#         'forum_link': f"https://openreview.net/forum?id={submission.id}",
#         'pdf_link': f"https://openreview.net/pdf?id={submission.id}"
#     }
#     return submission_info



# def contains_text(submission: dict, target_text: str, fields: Union[str, List[str]] = ['title', 'abstract'], is_regex: bool = False) -> bool:
#     # If 'all', consider all available keys in the submission for matching
#     if fields == 'all':
#         fields = ['title', 'abstract', 'keywords', 'primary_area', 'TLDR']

#     # Convert string input for fields into a list
#     if isinstance(fields, str):
#         fields = [fields]
    
#     # Iterate over the specified fields
#     for field in fields:
#         content = submission.get(field, "")
        
#         # Join lists into a single string (e.g., keywords)
#         if isinstance(content, list):
#             content = " ".join(content)
        
#         # Check if the target_text is found in the content of the field
#         if is_regex:
#             if re.search(target_text, content):
#                 return True
#         else:
#             if target_text in content:
#                 return True
    
#     # If no matches were found, return False
#     return False


# def search_submissions(submissions: List[Dict], target_text: str, fields: Union[str, List[str]] = ['title', 'abstract'], is_regex: bool = False) -> List[Dict]:
#     """
#     Search through the list of submissions and return those that match the target text.
    
#     :param submissions: List of submission dictionaries to search through.
#     :param target_text: The text to search for in each submission.
#     :param fields: The fields to search within for matching. Default is ['title', 'abstract'].
#     :param is_regex: Boolean flag indicating whether to use regex for matching. Default is False.
#     :return: List of submissions matching the target text.
#     """
#     # List to hold matching submissions
#     matching_submissions = []
    
#     for submission in submissions:
#         if contains_text(submission, target_text, fields, is_regex):
#             matching_submissions.append(submission)
    
#     return matching_submissions

# def write_to_csv(submissions: List[Dict], output_file: str):
#     """
#     将匹配的提交写入CSV文件，包括标题、摘要和链接。

#     :param submissions: 要写入的提交字典列表。
#     :param output_file: 输出CSV文件的路径。
#     """
#     with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['Title', 'Abstract', 'Forum Link', 'PDF Link']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for sub in submissions:
#             writer.writerow({
#                 'Title': sub.get('title', ''),
#                 'Abstract': sub.get('abstract', ''),
#                 'Forum Link': sub.get('forum_link', ''),
#                 'PDF Link': sub.get('pdf_link', '')
#             })


# if __name__ == "__main__":
#     # 初始化客户端
#     client = openreview.api.OpenReviewClient(
#         baseurl='https://api2.openreview.net',
#         username='3521403823@tju.edu.cn',
#         password='Mjx123456!'
#     )

#     # 获取论文列表
#     venue_id = 'ICLR.cc/2025/Conference'
#     all_submissions = get_submissions(client, venue_id)
#     submissions = get_submissions(client, venue_id, 'under_review')

#     # 提取论文数据
#     submission_infos = [extract_submission_info(sub) for sub in submissions]

#     # 检索关键词
#     langs = ['code agent']
#     lang_regex = '|'.join(langs)
#     matching_submissions = search_submissions(submission_infos, lang_regex, is_regex=True, fields='all')
#     # for mat in matching_submissions:
#     #     print(mat['title'])

#     # 将匹配的论文写入CSV文件
#     output_csv = 'matching_papers.csv'
#     write_to_csv(matching_submissions, output_csv)
#     print(f"已将匹配的论文信息保存到 {output_csv}。")

# import re
# from typing import List
# from da_agent.agent.models import call_llm

# def generate_executable_code(tool_codes: dict) -> str:
#         """
#         根据工具描述和代码，调用 LLM 生成可执行的 Python 程序。
#         Args:
#             tools_description (str): 工具的描述。
#             tool_codes (dict): 工具的代码字典，键为工具名称，值为代码。
#         Returns:
#          str: LLM 生成的代码。
#         """
#         prompt = f"""
#             根据以下工具描述和代码，编写一个完整的可执行 Python 程序来调用这些工具并展示其功能。

#             ### 工具代码:
#             {tool_codes}

#             请确保生成的代码能够正确调用这些工具。
#         """

#         payload = {
#             "model": "gpt-4o",
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ],
#             "max_tokens": 1000,
#             "temperature": 0
#         }

#         print("Sending prompt to LLM for code generation...")

#         # 调用 LLM 生成代码
#         success, llm_response = call_llm(payload)
#         if success:
#             print(f"Generated response from LLM:\n{llm_response}")

#             # 提取代码块
#             extracted_code = extract_code_from_response(llm_response)
#             if extracted_code:
#                 print(f"Extracted executable code:\n{extracted_code}")
#                 return extracted_code
#             else:
#                 return "Failed to extract executable code from LLM response."
#         else:
#             return f"Failed to generate code: {llm_response}"
# def extract_code_from_response(llm_response: str) -> str:
#         """
#         从 LLM 的响应中提取 Python 代码块。

#         Args:
#             llm_response (str): LLM 生成的响应内容。

#         Returns:
#             str: 提取出的 Python 代码。如果未找到代码块，则返回空字符串。
#         """
#         # import re

#         # 正则表达式匹配 Python 代码块
#         code_pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
#         match = code_pattern.search(llm_response)
#         if match:
#             return match.group(1).strip()
#         return ""

# # 示例用法



# tool_codes = {
#     "view_file_head": """
# def view_file_head(file_name: str, num_lines: int) -> None:
#     '''Prints the first few lines of a file for inspection.'''
#         with open(file_name, 'r') as file:
#             for _ in range(num_lines):
#                 print(file.readline().strip())
#     """
# }

# # 调用函数生成可执行代码
# executable_code = generate_executable_code(tool_codes)
# print(executable_code)
# print(extract_code_from_response(executable_code))

# import os
# import json
# from glob import glob
# from typing import List, Set, Dict, Union
# from da_agent.agent.agents import PromptAgent
# from transformers.agents.agents import Toolbox
# from da_agent.agent.generatedtool import GeneratedTool, parse_generated_tools

# def jaccard_similarity(desc1: str, desc2: str) -> float:
#     """计算两个描述之间的Jaccard相似度"""
#     set1: Set[str] = set(desc1.lower().split())
#     set2: Set[str] = set(desc2.lower().split())
#     intersection = set1.intersection(set2)
#     union = set1.union(set2)
#     if not union:
#         return 0.0
#     return len(intersection) / len(union)

# def find_similar_tools(tools: List[GeneratedTool], threshold: float = 0.5) -> List[List[int]]:
#     """
#     查找相似的工具，返回工具索引的组合列表
#     使用Jaccard相似度作为相似性度量
#     """
#     similar_groups: List[List[int]] = []
#     visited: Set[int] = set()

#     for i in range(len(tools)):
#         if i in visited:
#             continue
#         group = [i]
#         visited.add(i)
#         for j in range(i + 1, len(tools)):
#             if j in visited:
#                 continue
#             sim = jaccard_similarity(tools[i].description, tools[j].description)
#             if sim >= threshold:
#                 group.append(j)
#                 visited.add(j)
#         if len(group) > 1:
#             similar_groups.append(group)
#     return similar_groups

# def delete_tools(agent: PromptAgent, toolbox: any, tools_to_delete: List[int]):
#     """
#     删除指定索引的工具，包括从工具箱和文件系统中删除
#     """
#     for index in sorted(tools_to_delete, reverse=True):
#         tool = toolbox.tools_list[index]
#         tool_name = tool.name
#         print(f"删除相似工具: {tool_name}")

#         # 从工具箱中移除工具
#         try:
#             agent.generated_toolbox.remove_tool(tool_name)
#             print(f"已从工具库中移除工具: {tool_name}")
#         except KeyError:
#             print(f"工具库中不存在工具: {tool_name}")

#         # 删除对应的 .py 文件
#         existing_tool_files = glob(os.path.join(agent.generated_tool_dir, f"*_{tool_name}.py"))
#         for file_path in existing_tool_files:
#             try:
#                 os.remove(file_path)
#                 print(f"已删除工具文件: {file_path}")
#             except Exception as e:
#                 print(f"删除文件 '{file_path}' 时发生错误: {e}")

# def renumber_tools(agent: PromptAgent, toolbox: any):
#     """
#     重新编号工具文件，确保编号连续
#     """
#     tools = toolbox.tools_list
#     tools_sorted = sorted(tools, key=lambda x: x.name)  # 按名称排序，或根据其他规则排序
#     for idx, tool in enumerate(tools_sorted, start=1):
#         new_file_name = f"{str(idx).zfill(4)}_{tool.name}.py"
#         old_files = glob(os.path.join(agent.generated_tool_dir, f"*_{tool.name}.py"))
#         if old_files:
#             old_file_path = old_files[0]  # 假设只有一个文件
#             new_file_path = os.path.join(agent.generated_tool_dir, new_file_name)
#             if old_file_path != new_file_path:
#                 try:
#                     os.rename(old_file_path, new_file_path)
#                     print(f"已重命名文件: {old_file_path} -> {new_file_path}")
#                 except Exception as e:
#                     print(f"重命名文件 '{old_file_path}' 时发生错误: {e}")

# def load_generated_tools(agent: PromptAgent) -> any:
#     """
#     加载生成的工具并返回工具箱对象
#     """
#     generated_tools: List[GeneratedTool] = []
#     generated_tool_paths = sorted(glob(os.path.join(agent.generated_tool_dir, "*.py")))
#     for path in generated_tool_paths:
#         with open(path, "r", encoding="utf-8") as f:
#             code = f.read()
#         tools = parse_generated_tools(code)
#         generated_tools.extend(tools)
#     return Toolbox(generated_tools)  # 确保 Toolbox 类能够接受生成的工具列表

# def main():
#     # 初始化 PromptAgent
#     agent = PromptAgent(
#         model="gpt-4o",
#         max_tokens=1000,
#         top_p=0.9,
#         temperature=0,
#         max_memory_length=1500,
#         max_steps=20,
#         generated_tool_dir="generated_actions"
#     )

#     # 加载生成的工具
#     toolbox = load_generated_tools(agent)

#     # 将工具列表转换为有序列表以便通过索引访问
#     toolbox.tools_list = list(toolbox.tools.values())

#     # 查找相似的工具
#     similar_groups = find_similar_tools(toolbox.tools_list, threshold=0.5)

#     if similar_groups:
#         tools_to_delete = []
#         for group in similar_groups:
#             # 保留第一个工具，删除其余相似工具
#             tools_to_delete.extend(group[1:])
#         delete_tools(agent, toolbox, tools_to_delete)

#         # 重新编号剩余的工具文件
#         renumber_tools(agent, toolbox)

#         # 保存更新后的工具库
#         agent.save_generated_tools("")  # 传入空字符串或适当的参数以保存工具库
#     else:
#         print("未找到相似的工具，无需删除。")

#     # 打印当前工具名称
#     current_tools = toolbox.tools
#     print("当前工具箱中的工具:")
#     for tool_name in current_tools:
#         print(f"- {tool_name}")

#     print(toolbox)

# if __name__ == "__main__":
#     main()


# import os
# from da_agent.agent.tool_retriever import ToolRetriever  # 根据实际项目的模块结构调整导入路径

# def inspect_tool_retriever():
#     # 设置 `generated_tool_dir` 的路径
#     generated_tool_dir = "generated_actions"  # 替换为实际路径
#     query = "list files"  # 替换为需要查询的内容

#     # 初始化 ToolRetriever
#     tool_retriever = ToolRetriever(generated_tool_dir)

#     # 查看 self.generated_tools 中的工具
#     print("=== Registered tools in self.generated_tools ===")
#     for tool_name in tool_retriever.generated_tools.keys():
#         print(f" - {tool_name}")

#     # 调用 retrieve 方法，并查看返回结果
#     print("\n=== Results from retrieve method ===")
#     retrieved_tools = tool_retriever.retrieve(query, k=5)  # 根据需要调整 k 的值
#     if not retrieved_tools:
#         print("No tools retrieved.")
#     else:
#         for tool in retrieved_tools:
#             print(f"Retrieved tool: {tool.name}, Description: {tool.description}")

#     # 查看 similarity_search_with_score 返回的结果结构
#     print("\n=== Debug similarity_search_with_score ===")
#     docs_and_scores = tool_retriever.vectordb.similarity_search_with_score(query, k=5)  # 根据需要调整 k 的值
#     for doc, score in docs_and_scores:
#         print(f"Metadata: {doc.metadata}, Score: {score}")

# if __name__ == "__main__":
    # inspect_tool_retriever()

def list_files_in_directory() -> list:
    """Returns a list of files in the current directory."""
    import os
    return os.listdir('.')

def main():
    # Call the function to list files
    files = list_files_in_directory()

    # Print the list of files
    if files:
        print("Files in the current directory:")
        for file in files:
            print(file)
    else:
        print("No files found in the current directory.")

if __name__ == "__main__":
    main()