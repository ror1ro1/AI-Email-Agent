import os
import json
from google import genai
from google.genai import types

# Inicjalizacja oficjalnego klienta Gemini API
client = genai.Client()

# Symulacja skrzynki odbiorczej (surowe maile od klientów)
incoming_emails = [
    {
        "sender": "klient.hurt@example.com",
        "subject": "Zapytanie o cennik hurtowy",
        "body": "Dzień dobry, chciałbym otrzymać Państwa aktualny cennik hurtowy na systemy automatyzacji dla 50 stanowisk."
    },
    {
        "sender": "angry.user@example.com",
        "subject": "Problem z logowaniem do platformy",
        "body": "Wasz system znowu nie działa! Od godziny próbuję się zalogować i ciągle wyskakuje błąd serwera 500. Żądam naprawy."
    }
]

# Funkcja przetwarzająca e-mail za pomocą agenta AI
def process_email_with_agent(email_data):
    prompt = f"""
    Analyze the incoming corporate email. Detect the intent (Sales, Support, Spam), 
    determine the emotional tone (Positive, Neutral, Negative), and draft a highly professional, 
    context-aware response in the same language as the incoming mail.
    
    Email Sender: {email_data['sender']}
    Email Subject: {email_data['subject']}
    Email Body: {email_data['body']}
    """

    # Definiowanie sztywnej struktury wyjściowej JSON dla agenta workflow
    class EmailAgentOutput(types.BaseModel):
        detected_intent: str
        emotional_tone: str
        internal_routing_department: str
        draft_response: str

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EmailAgentOutput,
            temperature=0.2
        ),
    )
    
    return json.loads(response.text)

def main():
    print("Uruchamianie Agenta AI do analizy i automatyzacji poczty...\n")
    
    for idx, email in enumerate(incoming_emails, 1):
        print(f"--- Przetwarzanie wiadomości #{idx} od: {email['sender']} ---")
        try:
            agent_decision = process_email_with_agent(email)
            
            print(f"Wykryta intencja: {agent_decision['detected_intent']}")
            print(f"Wydźwięk emocjonalny: {agent_decision['emotional_tone']}")
            print(f"Kierowanie wewnętrzne do: {agent_decision['internal_routing_department']}")
            print("Wygenerowana wersja robocza odpowiedzi:")
            print(f"\"{agent_decision['draft_response']}\"")
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"Błąd podczas przetwarzania wiadomości przez Agenta: {e}\n")

if __name__ == "__main__":
    main()
