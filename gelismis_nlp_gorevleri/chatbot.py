from urllib import response
import openai

openai.api_key = "api_key"

def chat_with_gpt(promt, history_list):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Bu bizim mesajımız: {promt}. Konuşma geçmişi: {history_list}"}]
        )

    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    
    history_list = []
    while True:
        user_input = input("Kullanıcı tarafondan girilen mesaj: ")
        
        if user_input.lower() in ["exit", "q", "quit"]:
            print("Sohbet sonlandırıldı.")
            break
        history_list.append(user_input)
        response = chat_with_gpt(user_input, history_list)
        print(f"Chatbot: {response}")