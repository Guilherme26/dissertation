import logging

def get_toxicity_score(service, text, toxicity_type="TOXICITY"):
    toxicity_score = None

    try:
        if len(text) > 0 and len(text) < 3000:
            analyze_request = {
                'comment': {'text': text},
                'requestedAttributes': {toxicity_type: {}}
            }
            response = service.comments().analyze(body=analyze_request).execute()
            toxicity_score = (
                response.get("attributeScores")
                .get(toxicity_type)
                .get("summaryScore")
                .get("value")
            )
    except Exception as e:
        logging.error(f"The following error occured: \n{e.args}")
    
    return toxicity_score
