from typing import Dict

class PromptTemplate:
    def __init__(self, template_config: Dict):
        self.template = template_config
        
    def render(self, context: Dict) -> str:
        rendered = {}
        for key, value in self.template.items():
            if isinstance(value, tuple):
                rendered[key] = "\n".join([part.format(**context) for part in value])
            else:
                rendered[key] = value.format(**context)
        return self._format_prompt(rendered)
    
    def _format_prompt(self, fields: Dict) -> str:
        return "\n".join([f"{key}: {value}" for key, value in fields.items()])
    
class ActionListTemplate(PromptTemplate):
    DEFAULT_CONFIG = {
        "Query": "{query}",
        "Instructions": "Given a list of actions with action names: descriptions and a user query, determine the optimal sequence of actions to resolve the query. Simple query = simple action. Only include the necessary actions in the final response, and the last action must be action to response to user",
        "ActionList": "{action_list}",
        "Output": "Return a list contains ONLY action names seperated by '/'. No rationale or explanation is needed.",
        # "OutputFormat": "action1/action2/action3. Dont include action descriptions"
    }
    
    def __init__(self):
        super().__init__(self.DEFAULT_CONFIG)
        
    
class ContextAnalyserTemplate(PromptTemplate):
    DEFAULT_CONFIG = {
        "Query": "{query}",
        "Instructions": "Given user query and context retrieved from the database, generate a refined context that is relevant to the query for another LLM to use. Make sure to keep relevant information such as formulas, tables, etc, refine table if possible and remove irrelevant information.",
        "context": "{context}"
    }
    
    def __init__(self):
        super().__init__(self.DEFAULT_CONFIG)
        
    
class SimpleResponseTemplate(PromptTemplate):
    DEFAULT_CONFIG = {
        "Query": "{query}",
        "Instructions": "Given user query and chat history. Response appropriately to the query.",
        "History": "{history}"
    }
    
    def __init__(self):
        super().__init__(self.DEFAULT_CONFIG)
        
    
class ContextResponseTemplate(PromptTemplate):
    DEFAULT_CONFIG = {
        "Query": "{query}",
        "Instructions": "Given user query, context and chat history. Response appropriately to the query using the retrieved context from research papers. If the user only ask to show pdf, return nothing.",
        "Context": "{context}",
        "History": "{history}"
    }
    
    def __init__(self):
        super().__init__(self.DEFAULT_CONFIG)
        
    
class ContextTemplate(PromptTemplate):
    DEFAULT_CONFIG = {
        "Relevant Score": "{score}",
        "Title": "{title}",
        "Structure": "{structure}",
        "Text": "{text}"
    }
    
    def __init__(self):
        super().__init__(self.DEFAULT_CONFIG)
        