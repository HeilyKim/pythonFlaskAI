from selenium import webdriver
import time
from selenium.webdriver.common.by import By

brow = webdriver.Chrome()
brow.get(url='http://naver.com')
time.sleep(5)
print('test')
try:
    elem = brow.find_element(By.CLASS_NAME,'service_name')
    elem.click()
    brow.back()
    brow.forward()
    brow.refresh()
except Exception as e:
    print(e)
finally:
    time.sleep(5)
    brow.close()
    brow.quit()
