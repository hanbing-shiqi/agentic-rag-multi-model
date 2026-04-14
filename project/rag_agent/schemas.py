from typing import List, Optional
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    # 核心修改：改为 Optional[str]，并赋予默认值空字符串
    clarification_needed: Optional[str] = Field(
        default=None,
        description="Explanation if the question is unclear. Leave null if clear."
    )