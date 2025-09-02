import torch

def preprocess_image(pil_image,transform):
    transformed_image=transform(pil_image)
    return transformed_image.unsqueeze(0)


def predict_image(model, img_tensor, classes, device):
    model.eval()
    with torch.inference_mode():
        out = model(img_tensor.to(device))
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(out.argmax(1))
    return classes[idx], float(probs[idx]), probs
