from .usrConfig import PedanticClass

llmAgentType = "a self-evaluating agent"

# Singleton instance
pedantic_instance = PedanticClass(
                                  verbose=1, 
                                  retrieval=None,
                                  RLAIF = True,
                                  ShortMemory=False,
                                  LongMemory=False, 
                                  
                                  DocType="Complex Document",
                                  
                                  llmTask = """ 
                                  Your task is to generate an accurate answer based on the provided question and document. 
                                  Evaluate your answer against the question and RFP document for quality, completeness, and alignment. 
                                  If the generated answer is incomplete, inaccurate, or misaligned, 
                                  refine it to improve accuracy, detail, and completeness by reanalyzing the document.
                                  Ensure the answer is structured and does not alter any actual numbers, dates, names, or specific values from the document.
                                  If the requested information is not available in the document, you must explicitly state 'No Information available'.""",
                                  
                                  OutputFormat = """Use the provided context to generate a complete and detailed answer.
                                  If the context does not contain information relevant to the question, respond with 'No Information available'.
                                  Ensure the final answer includes all key points in bullets."""
                            
                                 )