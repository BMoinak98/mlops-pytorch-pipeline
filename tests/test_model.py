from src.model import get_model

def test_model_output_shape():
    model = get_model("resnet18", 10)
    assert model is not None