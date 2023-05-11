def preprocess_teacher(model, teacher):
    for param_m, param_t in zip(model.parameters(), teacher.parameters()):
        param_t.data.copy_(param_m.data)
        param_t.requires_grad = False