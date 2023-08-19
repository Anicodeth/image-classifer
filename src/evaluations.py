def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator)
    print("Test accuracy:", test_acc)
