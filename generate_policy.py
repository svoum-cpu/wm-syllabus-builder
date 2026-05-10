import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup & Config
load_dotenv()
CHROMA_PATH = "chroma_db"

# This is your specialized Professor-to-Policy Template
PROMPT_TEMPLATE = """
YOU ARE: An expert Academic Policy Advisor for the William & Mary School of Computing, Data Sciences & Physics. 

CONTEXT FROM CURRICULUM & HONOR CODE:
{context}

PROFESSOR'S INPUTS:
- Class Name: {class_name}
- Assignment Name: {assignment_name}
- Assignment Details: {assignment_details}
- Weight of Assignment: {weight}
- Learning Objectives: {learning_objectives}

TASK:
Generate a precise, assignment-specific AI-Use Policy. Consider the student's seniority based on the Class Name and the indexed curriculum. If the assignment is high-weight (e.g., >20%), the "Unacceptable" tier should be more rigorous.

OUTPUT FORMAT (Copy-Paste Ready):

### 🛡️ AI-Use Policy for {assignment_name}

**1. Acceptable Uses (Green Light):**
[Specific examples of how AI can assist without compromising the learning objectives.]

**2. Semi-Acceptable Uses (Yellow Light - Proceed with Caution):**
[Uses that require specific citation, a "reflection note" on how AI was used, or uses only permitted for debugging/explanation.]

**3. Unacceptable Uses (Red Light):**
[Strictly prohibited uses that would constitute a violation of the W&M Honor Code or bypass the core learning objectives of this task.]
"""

def main():
    # --- PROFESSOR INPUT FIELDS ---
    print("--- 🏛️ W&M AI Policy Generator ---")
    class_name = input("Class Name (e.g. DATA 201): ")
    assignment_name = input("Assignment Name: ")
    assignment_details = input("Assignment Details (Paste prompt): ")
    weight = input("Weight of Assignment (e.g. 15%): ")
    learning_objs = input("Learning Objectives: ")

    # 2. Connect to the Librarian's Brain (ChromaDB)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 3. Search for context (Finding the Class Curriculum & Honor Code rules)
    # We search using the Class Name and 'Honor Code' as the query
    search_query = f"{class_name} curriculum prerequisites and William & Mary Honor Code"
    results = db.similarity_search(search_query, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # 4. Fill the Template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        class_name=class_name,
        assignment_name=assignment_name,
        assignment_details=assignment_details,
        weight=weight,
        learning_objectives=learning_objs
    )

    # 5. Generate the Policy with Gemini
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))
    response = model.invoke(prompt)

    # 6. Output - THE ULTIMATE CLEANUP
    print("\n" + "="*30)
    print("✨ GENERATED POLICY ✨")
    print("="*30)
    
    final_text = ""

    # Strategy A: Check for standard .content string
    if hasattr(response, 'content') and isinstance(response.content, str):
        final_text = response.content
    
    # Strategy B: Handle the "List of Dictionaries" format we saw in your error
    elif isinstance(response.content, list) and len(response.content) > 0:
        if 'text' in response.content[0]:
            final_text = response.content[0]['text']
    
    # Strategy C: Fallback to the raw response if all else fails
    else:
        final_text = str(response)

    # Final Print - This should now be pure text
    print(final_text)

# The Ignition Switch
if __name__ == "__main__":
    main()