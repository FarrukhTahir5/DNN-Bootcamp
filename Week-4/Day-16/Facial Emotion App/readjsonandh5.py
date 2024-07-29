import json
from tensorflow.keras.models import model_from_json

# Load JSON file
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    print(loaded_model_json)
"""
# Create model from JSON
model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights('model_weights.h5')

# Compile the model (optional but recommended)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
"""