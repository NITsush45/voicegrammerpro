import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def score_grammar(text):
    matches = tool.check(text)
    num_errors = len(matches)
    score = max(0, 100 - num_errors * 5)

    result = []
    for match in matches:
        result.append({
            "error": text[match.offset : match.offset + match.errorLength],
            "message": match.message,
            "suggestions": match.replacements
        })

    return score, result
