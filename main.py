import tensorflow as tf
from src.data_processing import get_data_generators
from src.model import create_model
from src.evaluations  import evaluate_model

# Define data paths and other constants
train_data_dir = 'data/train'
validation_data_dir = 'data/validate'
test_data_dir = 'data/test'
num_classes = 5
batch_size = 32
epochs = 10

# Get data generators
train_generator, validation_generator, test_generator = get_data_generators(
    train_data_dir, validation_data_dir, test_data_dir, batch_size
)

# Create and compile the model
model = create_model(num_classes)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs
)

# Evaluate the model on the test set
evaluate_model(model, test_generator)

# Save the model
model.save('models/animals_model.h5')
