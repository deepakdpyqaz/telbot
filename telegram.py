import requests
import os

TELEGRAM_TOKEN = "5765471758:AAFPzn2Z2gbbe0sp6yurqxwbSmYrrGanla4"
TELEGRAM_CHAT_ID = "1479006629"

def send_to_telegram(text):
    '''
    Sends a text to telegram chat
    text: text to send
    '''
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    try:
        params = {'chat_id': chat_id, 'text': text}
        response = requests.post(url, params=params)
        return response
    except Exception as e:
        print(e)
        return False

def send_to_telegram_document(document):
    '''
    Sends a document to telegram chat
    document: path to file
    '''
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    url = f'https://api.telegram.org/bot{token}/sendDocument'
    params = {'chat_id': chat_id}
    post_files = {'document': open(document, 'rb')}
    header_list = {
        "Accept":"*/*",
        "User-Agent":"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0"
    }
    try:
        response = requests.post(url, params=params, data="", headers=header_list, files=post_files)
        return response
    except Exception as e:
        print(e)
        return False

