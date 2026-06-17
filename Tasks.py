from .config_instance import pedantic_instance


def getGeneratedQuestionTask():
    generateQuestionTask = f"""You are a {pedantic_instance.DocType} Question Rewriter, specialized in simplifying and rephrasing complex questions. 
                        Rewrite the given input question in an easy-to-understand way. 
                        Ensure that all critical terms, domain-specific phrases, and key context from the original question are explicitly retained in the rewritten question. 
                        Avoid removing or altering any important terminology or context. Respond with only the rewritten question, without any additional text, commentary, or
                        explanation."""
    return generateQuestionTask


def getContextEvaluationTask(doctype):
    contextEvaluationTask = f""""You are a highly skilled {doctype} evaluator. Assess whether the provided context sufficiently answers the question.
                           "If it does, respond with 'Sufficient context'. If not, respond with 'Insufficient context'."""
    return contextEvaluationTask



def getFinalAnswerTask():
    generateFinalAnswerTask = pedantic_instance.llmTask
    return generateFinalAnswerTask

def getOuputFormatTask():
    generateFinalAnswerTask = pedantic_instance.OutputFormat
    return generateFinalAnswerTask

