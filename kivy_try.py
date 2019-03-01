import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
class App_try(App):

    def build(self):
        login_face = my_login()

        return login_face


class my_login(GridLayout):
    def __init__(self,**kwargs):
        super(my_login, self).__init__(**kwargs)
        self.cols = 2
        self.add_widget(Label(text = 'Please input your name'))
        self.face_name =  TextInput(multiline = False)
        self.add_widget(self.face_name)
        self.add_widget(Label(text = 'Please input your age'))
        self.Age = TextInput(multiline = False)
        self.add_widget(self.Age)




print(index)

if __name__ =='__main__':
    app = App_try()
    app.run()


