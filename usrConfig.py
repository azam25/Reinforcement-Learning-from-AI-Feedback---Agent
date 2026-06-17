from pydantic import BaseModel, Field

class PedanticClass(BaseModel):
    verbose: int = Field(..., description="An integer representing verbosity level")
    DocType: str = Field(..., description="A string representing the document type")
    llmTask: str = Field(..., description="A string representing the question task")
    OutputFormat: str = Field(..., description="A string representing the document type")
    retrieval: object = Field(..., description="An object for general retrieval functionality")
    RLAIF: bool = Field(..., description="An bool value representing LLM working method")
    ShortMemory: bool = Field(..., description="An bool value representing context Memory type")
    LongMemory: bool = Field(..., description="An bool value representing context Memory type")