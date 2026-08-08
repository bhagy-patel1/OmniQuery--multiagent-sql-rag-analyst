from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    sql_error: Optional[str]
    sql_retry_count: int
    sql_validation_error: Optional[str]
    execution_status: str  # "success", "retry", "failed", "blocked"
    final_response: Optional[str]  # Human-readable answer produced by Responder