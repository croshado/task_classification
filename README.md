# **Heuristic-Based Task Extraction System**

## **Introduction**
This document explains the working of the heuristic-based task extraction system. The program is designed to analyze unstructured text, identify tasks, extract the assignees, and determine deadlines if available. It follows a rule-based approach without using any large language models (LLMs) or pre-trained models, relying instead on NLP techniques and heuristics.

## **Workflow Overview**
The system processes input text and performs the following steps:
1. **Preprocessing**: Cleans the text and tokenizes it into sentences.
2. **Task Extraction**: Identifies sentences that represent actionable tasks based on heuristic rules.
3. **Entity Recognition**: Extracts assignees (who has to do the task) and deadlines (when it is due).
4. **Output Generation**: Produces a structured list of tasks with assigned personnel and deadlines.

## **Code Explanation**

### **1. Import Required Libraries**
```python
import re
import spacy
```
- `re` is used for text preprocessing (removing unnecessary characters).
- `spacy` is used for natural language processing (NLP) tasks like tokenization and named entity recognition (NER).

### **2. Load NLP Model**
```python
nlp = spacy.load("en_core_web_sm")
```
- Loads the `en_core_web_sm` model, which provides functionalities like POS tagging and named entity recognition.

### **3. Define Sample Text Input**
```python
text = """
John needs to review the project report by Monday. Alice should schedule a meeting for the Q1 budget plan by Friday.
Please submit the final design to Mark before the end of the week.
"""
```
- This is the input text that contains multiple task statements.

### **4. Define Action Verbs**
```python
action_verbs = {"review", "schedule", "submit", "discuss", "approve", "analyze", "check", "send", "deliver"}
```
- These are verbs that indicate an action or a task. The program checks for their presence in sentences to identify tasks.

### **5. Preprocess the Text**
```python
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)  # Remove punctuation except periods
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]
```
- Cleans the text by removing unnecessary punctuation.
- Splits the text into sentences.

### **6. Extract Tasks, Assignees, and Deadlines**
```python
def extract_tasks(sentences):
    tasks = []
    for sent in sentences:
        doc = nlp(sent)
        words = {token.text.lower() for token in doc}
        if words & action_verbs:
            task_info = {"task": sent, "assignee": None, "deadline": None}
            assignees = [token.text for token in doc if token.pos_ == "PROPN"]
            if assignees:
                task_info["assignee"] = assignees[0]
            deadlines = [ent.text for ent in doc.ents if ent.label_ in {"DATE", "TIME"}]
            if deadlines:
                task_info["deadline"] = deadlines[0]
            tasks.append(task_info)
    return tasks
```
- Checks if the sentence contains an action verb.
- Extracts proper nouns as potential assignees.
- Identifies dates and time expressions as deadlines.

### **7. Process the Text and Print the Results**
```python
sentences = preprocess_text(text)
tasks = extract_tasks(sentences)
for task in tasks:
    print(task)
```
- Preprocesses the text.
- Extracts tasks.
- Prints structured task details.

## **Example Test Cases**
Here are some additional test cases to validate the program’s performance.

### **Test Case 1**
#### **Input:**
```text
Sarah needs to send the updated presentation to the client by tomorrow morning.
```
#### **Expected Output:**
```json
[

    {"task": "Sarah needs to send the updated presentation to the client by tomorrow morning.", "assignee": "Sarah", "deadline": "tomorrow morning"},

]
```

### **Test Case 2**
#### **Input:**
```text
David is required to submit the budget proposal by the end of the month.
Emma will review the project plan by next Monday.
```
#### **Expected Output:**
```json
[
    {"task": "David is required to submit the budget proposal by the end of the month.", "assignee": "David", "deadline": "end of the month"},
    {"task": "Emma will review the project plan by next Monday.", "assignee": "Emma", "deadline": "next Monday"},
]
```

### **Test Case 3**
#### **Input:**
```text
Lisa is supposed to check the inventory before the weekend.
Tom and Jerry need to discuss the marketing strategy by the end of this week.
Alice has to prepare the sales report and submit it to the director before Monday.
```
#### **Expected Output:**
```json
[
    {"task": "Lisa is supposed to check the inventory before the weekend.", "assignee": "Lisa", "deadline": "the weekend"},
    {"task": "Tom and Jerry need to discuss the marketing strategy by the end of this week.", "assignee": "Tom, Jerry", "deadline": "end of this week"},
    {"task": "Alice has to prepare the sales report and submit it to the director before Monday.", "assignee": "Alice", "deadline": "Monday"}
]
```

## **Conclusion**
This heuristic-based system efficiently extracts tasks, assignees, and deadlines from unstructured text using predefined rules. It can be further improved by incorporating:
- **A more extensive action verb dictionary** for better task recognition.
- **Better handling of multiple assignees and deadlines** in a single sentence.
- **Support for different text formats**, such as emails and meeting minutes.

This system provides a solid foundation for automating task extraction without relying on deep learning models, making it efficient and easy to deploy in various environments. 🚀

