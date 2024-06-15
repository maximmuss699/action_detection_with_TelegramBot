import telebot
import threading

TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
CHAT_ID = 123456789 # YOUR_CHAT_ID

bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f"Your chat ID is {message.chat.id}")

def start_bot():
    bot.polling(none_stop=True)

bot_thread = threading.Thread(target=start_bot)
bot_thread.daemon = True
bot_thread.start()

def send_notification(image_path, caption):
    with open(image_path, 'rb') as photo:
        bot.send_photo(CHAT_ID, photo, caption=caption)
