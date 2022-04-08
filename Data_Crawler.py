import mouse, keyboard
import time

def click_next_video():
    mouse.move(1632, 170)
    mouse.click()
    time.sleep(1)

def copy_link():
    mouse.move(504, 50)
    mouse.click()
    time.sleep(0.6)
    keyboard.send("ctrl+c")

if __name__ == '__main__':

    # while True:
    #    print(mouse.get_position())
    time.sleep(15) # time to change the window to Youtube

    for i in range(10):
        click_next_video()
        copy_link()
