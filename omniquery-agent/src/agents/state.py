from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    route: Optional[str]  
    
    # Session & User Database Configs
    session_id: Optional[str]
    user_db_config: Optional[Dict[str, Any]]

    # SQL Subgraph Memory
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    sql_error: Optional[str]
    sql_retry_count: int
    sql_validation_error: Optional[str]
    sql_status: Optional[str]

    # CSV Subgraph Memory
    csv_query: Optional[str]
    csv_result: Optional[List[Dict[str, Any]]]
    csv_error: Optional[str]

    # RAG Subgraph Memory
    rag_context: Optional[List[Dict[str, Any]]]
    rag_error: Optional[str]

    # Master Output
    final_response: Optional[str]