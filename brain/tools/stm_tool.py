"""
brain/tools/stm_tool.py
STM tool for LLM to access conversation history.
"""

class STMTool:
    def __init__(self, stm_module):
        self.stm = stm_module

    def get_conversation_history(self) -> str:
        """Get the recent conversation history."""
        ctx = self.stm.get_context()
        if not ctx:
            return "Conversation history is empty."
        return ctx
