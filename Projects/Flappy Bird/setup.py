from cx_Freeze import setup, Executable

# Replace 'your_script.py' with your script's filename
setup(
    name="AI Flappy Bird",
    version="1.0",
    description="Flappy Bird",
    executables=[Executable("C:/Users/Farrukh/DNN-Bootcamp/Projects/Flappy Bird/flappy_bird.py")]
)
