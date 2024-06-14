import telebot
from datetime import datetime
import threading


TELEGRAM_TOKEN = 'YOUR_TELEGRAM' # Your telegram token
CHAT_ID = 1234567890 # Your chat ID

bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f"Your chat ID is {message.chat.id}")

def start_bot():
    bot.polling(none_stop=True)

bot_thread = threading.Thread(target=start_bot)
bot_thread.daemon = True
bot_thread.start()

def send_notification(image_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    caption = f"Person detected at {current_time}"
    with open(image_path, 'rb') as photo:
        bot.send_photo(CHAT_ID, photo, caption=caption)
