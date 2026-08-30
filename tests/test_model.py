from src.model import get_model

def test_model_output_shape():
    model = get_model("resnet18", 10)
    assert model is not None
    
def test_model_architecture():
    model = get_model("resnet18", num_classes=10)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)