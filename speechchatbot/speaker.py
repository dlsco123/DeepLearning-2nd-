import time

import speech_recognition as sr
from gtts import gTTS
import playsound

while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('음성 인식')
        audio = r.listen(source)
        result = r.recognize_google(audio,language='ko')
        print('음성 : ' + result)

        if '카카오' in result:
            tts = gTTS(text='반갑습니다. 주인님', lang='ko')
            tts.save('test.mp3')
            time.sleep(3)
            playsound.playsound('test.mp3', True)



# import time
# import speech_recognition as sr
# from gtts import gTTS
# import playsound

# try:
# while True:```
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print('음성 인식')
#         audio = r.listen(source)
#         result = r.recognize_google(audio,language='ko')
#         print('음성 : ' + result)
#
#         if '카카오' in result:
#             tts = gTTS(text = '반갑습니다. 주인님', lang='ko')
#             filename = 'test.mp3'
#             tts.save(filename)
#             time.sleep(3)
#             playsound.playsound(filename)
# except Exception as e:
#     print(f"An error occurred: {e}")

