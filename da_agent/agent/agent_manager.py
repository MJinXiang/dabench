# # agent_manager.py

# import logging
# from typing import List, Dict, Any

# # 假设COTAgent和PromptAgent已经在agents.py中定义并导入
# from da_agent.agent.agents import PromptAgent
# from da_agent.agent.COT_agent import COTAgent

# logger = logging.getLogger("da_agent")

# class AgentManager:
#     def __init__(self, cot_agent: COTAgent, prompt_agent: PromptAgent):
#         self.cot_agent = cot_agent
#         self.prompt_agent = prompt_agent

#     def process_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         处理单个任务：首先使用COTAgent生成COT提示，然后使用PromptAgent生成最终响应。

#         :param task_config: 任务配置字典。
#         :return: 结果字典，包含COTAgent和PromptAgent的输出。
#         """
#         try:
#             task_id = task_config.get("id", "未知")
#             prompt = task_config.get("prompt", "请描述任务。")
#             # logger.info(f"开始处理任务: {task_id}")

#             # 使用COTAgent生成思考步骤（COT提示）
#             cot_output = self.cot_agent.generate_thoughts(prompt)
#             # logger.debug(f"COTAgent 输出: {cot_output}")

#             # 将COT提示作为系统消息传递给PromptAgent，并生成最终响应
#             final_response = self.prompt_agent.generate_response(cot_output)
#             # logger.debug(f"PromptAgent 输出: {final_response}")

#             result = {
#                 "task_id": task_id,
#                 "cot_output": cot_output,
#                 "final_response": final_response
#             }

#             # logger.info(f"任务完成: {task_id}")
#             return result

#         except Exception as e:
#             # logger.exception(f"处理任务时发生异常: {e}")
#             return {
#                 "task_id": task_config.get("id", "未知"),
#                 "error": str(e)
#             }

#     def process_tasks(self, task_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         处理多个任务。

#         :param task_configs: 任务配置列表。
#         :return: 结果列表。
#         """
#         results = []
#         for task in task_configs:
#             result = self.process_task(task)
#             results.append(result)
#         return results