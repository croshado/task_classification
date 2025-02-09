import re
import spacy

# Load Spacy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Sample text input
text = """
Lisa is supposed to check the inventory before the weekend.  
Tom and Jerry need to discuss the marketing strategy by the end of this week.  
Alice has to prepare the sales report and submit it to the director before Monday.  

"""

# List of action verbs that indicate a task
action_verbs = {"review", "schedule", "submit", "discuss", "approve", "analyze", "check", "send", "deliver"}

def preprocess_text(text):
    """Cleans and tokenizes the text."""
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)  # Remove punctuation except periods
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_tasks(sentences):
    """Extracts tasks, assignees, and deadlines using heuristics."""
    tasks = []
    
    for sent in sentences:
        doc = nlp(sent)
        words = {token.text.lower() for token in doc}
        
        # Check if the sentence contains an action verb
        if words & action_verbs:
            task_info = {"task": sent, "assignee": None, "deadline": None}
            
            # Identify assignee (Proper Noun appearing before the verb)
            assignees = [token.text for token in doc if token.pos_ == "PROPN"]
            if assignees:
                task_info["assignee"] = assignees[0]  # Take the first proper noun as the assignee
            
            # Identify deadlines (Dates, Time expressions)
            deadlines = [ent.text for ent in doc.ents if ent.label_ in {"DATE", "TIME"}]
            if deadlines:
                task_info["deadline"] = deadlines[0]  # Take the first recognized date/time
            
            tasks.append(task_info)
    
    return tasks

# Process text
sentences = preprocess_text(text)
tasks = extract_tasks(sentences)

# Print structured output
for task in tasks:
    print(task)
