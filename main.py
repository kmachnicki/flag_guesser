#!/usr/bin/env python3

from tkinter import *
from dataset import DataSet
from operator import itemgetter
from consts import WindowType, COLORS, QUESTIONS, QUESTIONS_ORDER,\
    TRAIN_SET_PATH, ALGORITHM, MAX_ITER, ALPHA, HIDDEN_LAYER_SIZE, RANDOM_STATE
from sklearn.neural_network import MLPClassifier


class Brain:
    def __init__(self, train_set_path, algorithm, max_iter, alpha, hidden_layer_size, random_state):
        self.ds = DataSet()
        with open(train_set_path, "r", newline='', encoding="utf8") as csv_file:
            self.ds.extract_from_csv(csv_file)

        self.clf = MLPClassifier(algorithm=algorithm,
                                 max_iter=max_iter,
                                 alpha=alpha,
                                 hidden_layer_sizes=hidden_layer_size,
                                 random_state=random_state)
        self.clf.fit(self.ds.X, self.ds.y)

        self.flag = []
        self.init_flag()

    def get_countries(self):
        return self.clf.classes_

    def init_flag(self):
        self.flag = []
        #print("DBG: Col names size: {}".format(len(self.ds.col_names)))
        #print("DBG: Flag size (should be 0): {}".format(len(self.flag)))
        for i in range(len(self.ds.col_names)):
            self.flag.append(0)
        #print("DBG: Flag size (should be equal to col names size): {}".format(len(self.flag)))

    def set_flag(self, index, value):
        self.flag[index] = value

    def predict(self, index):
        probabilities = self.clf.predict_proba([self.flag])
        countries_probabilities = {}
        #print("DBG: Classes size: {}".format(len(self.clf.classes_)))
        for k in range(len(self.clf.classes_)):
            countries_probabilities[self.clf.classes_[k]] = probabilities[0][k]

        sorted_probas = sorted(countries_probabilities.items(), key=itemgetter(1), reverse=True)
        #print("DBG: Index proba: {:.3f}% of country: {}".format(sorted_probas[index][1], sorted_probas[index][0]))
        #print("DBG: 1st proba: {:.3f}% of country: {}".format(sorted_probas[0][1], sorted_probas[0][0]))
        #print("DBG: 2nd proba: {:.3f}% of country: {}".format(sorted_probas[1][1], sorted_probas[1][0]))
        #print("DBG: 3rd proba: {:.3f}% of country: {}".format(sorted_probas[2][1], sorted_probas[2][0]))
        return sorted_probas[index][0]


class Pinky:
    def __init__(self, master, brain):
        self.master = master
        self.brain = brain

        self.frame = Frame(self.master)

        self.master.wm_title("Let's play a game")

        self.welcome_label = Label(self.frame)
        self.welcome_label["text"] = "Welcome stranger in this flag guesser game.\n" \
                                     "Please think about any flag you want, remember it, and I'll try to guess it.\n" \
                                     "Press Start when you are ready.\n"
        self.welcome_label.pack({"side": "top"})

        self.exit = Button(self.frame)
        self.exit["text"] = "Exit"
        self.exit["command"] = self.master.destroy
        self.exit.pack({"side": "left"})

        self.start = Button(self.frame)
        self.start["text"] = "Start"
        self.start["command"] = self.run_game
        self.start.pack({"side": "right"})

        self.frame.pack()

    def run_game(self):
        self.brain.init_flag()

        self.no_of_questions = 0
        self.discarded_countries = []

        if self.create_questions() == True:
            self.display_final_window(True)
            return

        for i in range(len(self.brain.get_countries())):
            predicted_country = self.brain.predict(i)
            if predicted_country not in self.discarded_countries:
                self.discarded_countries.append(predicted_country)
                if self.guess_country(i) == 1:
                    self.display_final_window(True)
                    break

        self.display_final_window(False)

    def display_final_window(self, guessed):
        self.new_window = Toplevel(self.master)
        self.app = FinalWindow(self.new_window, guessed, self.no_of_questions, len(self.discarded_countries))
        self.master.wait_window(self.new_window)

    def guess_country(self, index):
        self.new_window = Toplevel(self.master)
        country = self.brain.predict(index)
        self.app = FlagWindow(self.new_window, country)
        self.master.wait_window(self.new_window)
        return self.new_window.guessed

    def create_questions(self):
        for question_index in QUESTIONS_ORDER:
            if self.no_of_questions != 0 and self.no_of_questions % 6 == 0:
                for i in range(len(self.brain.get_countries())):
                    predicted_country = self.brain.predict(i)
                    if predicted_country not in self.discarded_countries:
                        self.discarded_countries.append(predicted_country)
                        if self.guess_country(i) == 1:
                            return True
                        break

            question = QUESTIONS[question_index]
            question_string = question[0]
            question_type = question[1]
            self.create_question_window(question_string, question_type, question_index)
            self.no_of_questions += 1
        return False

    def create_question_window(self, question_string, question_type, question_index):
        self.new_window = Toplevel(self.master)

        if question_type == WindowType.Boolean:
            self.app = BooleanWindow(self.new_window, self.brain, question_string, question_index)
        elif question_type == WindowType.Numeric:
            self.app = NumericWindow(self.new_window, self.brain, question_string, question_index)
        elif question_type == WindowType.Enumeration:
            self.app = EnumWindow(self.new_window, self.brain, question_string, question_index)

        self.master.wait_window(self.new_window)

class BooleanWindow:
    def __init__(self, master, brain, question, index):
        self.master = master
        self.brain = brain
        self.index = index
        self.frame = Frame(self.master)

        self.master.bind("<Return>", self.process_selected_value)
        self.master.bind("<Up>", self.scroll_up)
        self.master.bind("<Down>", self.scroll_down)

        self.master.wm_title("Question")

        self.welcome_label = Label(self.frame)
        self.welcome_label["text"] = question
        self.welcome_label.pack({"side": "top"})

        self.selected_value = IntVar()
        self.selected_value.set(0)

        self.radio_button_false = Radiobutton(self.frame)
        self.radio_button_false["text"] = "False"
        self.radio_button_false["value"] = 0
        self.radio_button_false["variable"] = self.selected_value
        self.radio_button_false["indicatoron"] = 0
        self.radio_button_false.select()
        self.radio_button_false.pack(anchor=CENTER)

        self.radio_button_true = Radiobutton(self.frame)
        self.radio_button_true["text"] = "True"
        self.radio_button_true["value"] = 1
        self.radio_button_true["variable"] = self.selected_value
        self.radio_button_true["indicatoron"] = 0
        self.radio_button_true.pack(anchor=CENTER)

        self.cont = Button(self.frame)
        self.cont["text"] = "Continue"
        self.cont["command"] = self.process_selected_value
        self.cont.pack({"side": "bottom"})

        self.frame.pack()

    def scroll_up(self, event=None):
        self.radio_button_false.select()

    def scroll_down(self, event=None):
        self.radio_button_true.select()

    def process_selected_value(self, event=None):
        selected_value = self.selected_value.get()
        #print("DBG: Boolean selected: {}".format(selected_value))
        self.brain.set_flag(self.index, selected_value)
        self.master.destroy()


class EnumWindow:
    def __init__(self, master, brain, question, index):
        self.master = master
        self.brain = brain
        self.index = index
        self.frame = Frame(self.master)

        self.master.bind("<Return>", self.process_selected_value)
        self.master.bind("<Up>", self.scroll_up)
        self.master.bind("<Down>", self.scroll_down)

        self.master.wm_title("Question")

        self.welcome_label = Label(self.frame)
        self.welcome_label["text"] = question
        self.welcome_label.pack({"side": "top"})

        self.listbox = Listbox(self.frame)
        for color, value in COLORS:
            self.listbox.insert(value, color)
        self.listbox["selectmode"] = SINGLE
        self.listbox.select_set(0)
        self.listbox.pack(anchor=CENTER)

        self.cont = Button(self.frame)
        self.cont["text"] = "Continue"
        self.cont["command"] = self.process_selected_value
        self.cont.pack({"side": "bottom"})

        self.frame.pack()

    def scroll_up(self, event=None):
        selected_index = self.listbox.curselection()[0]
        if selected_index != 0:
            self.listbox.select_clear(0, END)
            self.listbox.select_set(selected_index - 1)

    def scroll_down(self, event=None):
        selected_index = self.listbox.curselection()[0]
        if selected_index != len(self.listbox.get(0, END)) - 1:
            self.listbox.select_clear(0, END)
            self.listbox.select_set(selected_index + 1)

    def process_selected_value(self, event=None):
        selected_index = self.listbox.curselection()[0]
        selected_value = COLORS[selected_index][1]
        #print("DBG: Enum selected: {}".format(selected_value))
        self.brain.set_flag(self.index, selected_value)
        self.master.destroy()


class NumericWindow:
    def __init__(self, master, brain, question, index):
        self.master = master
        self.brain = brain
        self.index = index
        self.frame = Frame(self.master)

        self.master.bind("<Return>", self.process_selected_value)
        self.master.bind("<Up>", self.scroll_up)
        self.master.bind("<Down>", self.scroll_down)

        self.master.wm_title("Question")

        self.welcome_label = Label(self.frame)
        self.welcome_label["text"] = question
        self.welcome_label.pack({"side": "top"})

        self.listbox = Listbox(self.frame)
        for i in range(20):
            self.listbox.insert(i, i)
        self.listbox["selectmode"] = SINGLE
        self.listbox.select_set(0)
        self.listbox.pack(anchor=CENTER)

        self.cont = Button(self.frame)
        self.cont["text"] = "Continue"
        self.cont["command"] = self.process_selected_value
        self.cont.pack({"side": "bottom"})

        self.frame.pack()

    def scroll_up(self, event=None):
        selected_index = self.listbox.curselection()[0]
        if selected_index != 0:
            self.listbox.select_clear(0, END)
            self.listbox.select_set(selected_index - 1)

    def scroll_down(self, event=None):
        selected_index = self.listbox.curselection()[0]
        if selected_index != len(self.listbox.get(0, END)) - 1:
            self.listbox.select_clear(0, END)
            self.listbox.select_set(selected_index + 1)

    def process_selected_value(self, event=None):
        selected_value = self.listbox.curselection()[0]
        #print("DBG: Numeric selected: {}".format(selected_value))
        self.brain.set_flag(self.index, selected_value)
        self.master.destroy()


class FlagWindow:
    def __init__(self, master, flag):
        self.master = master
        self.master.guessed = False
        self.frame = Frame(self.master)

        self.master.bind("<Return>", self.process_selected_value)
        self.master.bind("<Up>", self.scroll_up)
        self.master.bind("<Down>", self.scroll_down)

        self.master.wm_title("Answer")

        self.welcome_label = Label(self.frame)
        self.welcome_label["text"] = "Is this {}?".format(flag.replace("-", " "))
        self.welcome_label.pack({"side": "top"})

        self.selected_value = IntVar()
        self.selected_value.set(0)

        self.radio_button_no = Radiobutton(self.frame)
        self.radio_button_no["text"] = "No"
        self.radio_button_no["value"] = False
        self.radio_button_no["variable"] = self.selected_value
        self.radio_button_no["indicatoron"] = 0
        self.radio_button_no.select()
        self.radio_button_no.pack(anchor=CENTER)

        self.radio_button_yes = Radiobutton(self.frame)
        self.radio_button_yes["text"] = "Yes"
        self.radio_button_yes["value"] = True
        self.radio_button_yes["variable"] = self.selected_value
        self.radio_button_yes["indicatoron"] = 0
        self.radio_button_yes.pack(anchor=CENTER)

        self.cont = Button(self.frame)
        self.cont["text"] = "Continue"
        self.cont["command"] = self.process_selected_value
        self.cont.pack({"side": "bottom"})

        self.frame.pack()

    def scroll_up(self, event=None):
        self.radio_button_no.select()

    def scroll_down(self, event=None):
        self.radio_button_yes.select()

    def process_selected_value(self, event=None):
        selected_value = self.selected_value.get()
        #print("DBG: Answer selected: {}".format(selected_value))
        self.master.guessed = selected_value
        self.master.destroy()


class FinalWindow:
    def __init__(self, master, guessed, no_of_questions, no_of_discarded_countries):
        self.master = master
        self.frame = Frame(self.master)

        self.master.bind("<Return>", self.process_selected_value)

        self.master.wm_title("Question")

        final_verdict = ""
        if guessed:
            final_verdict += "Ha! It took me {} questions and {} tries to guess the flag. " \
                             "Thanks for playing!\n".format(no_of_questions, no_of_discarded_countries)
        else:
            final_verdict += "I do not know what you're thinking about, and probably neither do you.\n" \
                             "Or maybe such country exists but in a parallel universe. Anyway, I tried.\n"

        final_verdict += "You can now press exit to finish the current game and begin a new one from the main menu."

        self.welcome_label = Label(self.frame)
        self.welcome_label["text"] = final_verdict

        self.welcome_label.pack({"side": "top"})

        self.cont = Button(self.frame)
        self.cont["text"] = "Exit"
        self.cont["command"] = self.process_selected_value
        self.cont.pack({"side": "bottom"})

        self.frame.pack()

    def process_selected_value(self, event=None):
        self.master.destroy()


def main():
    brain = Brain(TRAIN_SET_PATH, ALGORITHM, MAX_ITER, ALPHA, HIDDEN_LAYER_SIZE, RANDOM_STATE)
    root = Tk()
    pinky = Pinky(root, brain)
    root.mainloop()

if __name__ == "__main__":
    main()