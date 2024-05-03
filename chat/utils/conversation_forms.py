import pandas as pd

def load_conversation_form(form_id):
    # Load conversational form data from file
    form_data = pd.read_excel(f'data/forms/form{form_id}.xlsx')
    return form_data