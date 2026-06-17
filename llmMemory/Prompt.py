from . import Task
def getRefineQuestionPrompt(context,current_question):
    strPrompt = (
            f"Previous conversation:\n{context}\n"
            f"Current question: {current_question}\n"
        )
    strPrompt =  strPrompt +'\n' + Task.task

    return strPrompt

    