from streamlit_chat import message as stc
import openai

openai.api_key = 'pk-chOPDhlMdmAKqeGOeeMyOyMQqQLyIDRcgjKYzhHbWlmIuttK'
openai.api_base = 'https://api.pawan.krd/v1'

conv_history = [{"role" : "system",
     "content" : "Sos un asistente para seguridad vial. Vas a dar datos del clima cuando te pidan, tenes que dar Temperatura, Presion Atmosferica, Precipitaciones, Humedad, Velocidad del Viento, Sensacion Termica"}]


def display_chat_history(chat_history):
    for sender, message in chat_history:
        
        if sender == 'Usuario':
            stc(f"{message}", is_user=True)
        else:
            stc(f"{message}") 


def generate_bot_response(user_input):
    message = user_input
    
    conv_history.append({"role":"user", "content":message})
    
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo", messages=conv_history, max_tokens = 150, temperature = 0
    )

    response_content = response['choices'][0]['message']['content']

    if not response_content.endswith((".", "?", "!", "…")):
        conv_history.append({"role":"user", "content":'Continua con tu mensaje. Pero con un maximo de 100 palabras mas.'})
        response = openai.ChatCompletion.create(model = "gpt-3.5-turbo", messages=conv_history)
        response_content2 = response['choices'][0]['message']['content']
        f_response = response_content+response_content2
            

        conv_history.append({"role":"assistant", "content": response_content2})
        return f_response
    
    else:
        conv_history.append({"role":"assistant", "content": response_content})

        return response_content
