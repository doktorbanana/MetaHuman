import mouse, keyboard
import time

def click_next_video():
    mouse.move(504, 700)
    time.sleep(5)
    mouse.click()
    time.sleep(2)
    keyboard.send("shift+n")
    time.sleep(1)

def copy_link():
    mouse.move(504, 50)
    mouse.click()
    time.sleep(0.6)
    keyboard.send("ctrl+c")

if __name__ == '__main__':

    # while True:
    #    print(mouse.get_position())
    time.sleep(5) # time to change the window to Youtube

    for i in range(1000):
        if keyboard.is_pressed("space"):
            exit()
        copy_link()
        click_next_video()
        print(i)

