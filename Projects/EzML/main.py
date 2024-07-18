import tkinter as tk
from tkinter import Menu

def do_nothing():
    pass

def create_main_window():
    root = tk.Tk()
    root.title("Menu with Checkboxes")

    # Create main menu
    menu = Menu(root)
    root.config(menu=menu)

    # Create File menu
    file_menu = Menu(menu)
    menu.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Exit", command=root.quit)

    # Create Edit menu with Options submenu and checkboxes
    edit_menu = Menu(menu)
    menu.add_cascade(label="Edit", menu=edit_menu)

    options_menu = Menu(edit_menu)
    edit_menu.add_cascade(label="Options", menu=options_menu)

    # Add checkboxes
    var1 = tk.IntVar()
    var2 = tk.IntVar()
    options_menu.add_checkbutton(label="Option 1", variable=var1)
    options_menu.add_checkbutton(label="Option 2", variable=var2)

    root.mainloop()

if __name__ == "__main__":
    create_main_window()
