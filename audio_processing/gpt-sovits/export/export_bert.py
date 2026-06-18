import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer


def export_bert_onnx(model_name="hfl/chinese-roberta-wwm-ext-large", output_path="chinese-roberta.onnx"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    # Create wrapper that outputs hidden_states[-3]
    class BertWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            res = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )
            # Extract hidden_states[-3] (3rd from last layer)
            hidden_state = res["hidden_states"][-3]
            return hidden_state

    wrapper = BertWrapper(model)
    wrapper.eval()

    # Dummy input
    text = "你好世界"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    print(f"Input shape: {input_ids.shape}")

    # Export to ONNX
    print(f"Exporting to {output_path}")
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )
    print(f"Exported to {output_path}")

    # Verify
    import onnxruntime
    sess = onnxruntime.InferenceSession(output_path)
    onnx_out = sess.run(None, {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    })
    print(f"ONNX output shape: {onnx_out[0].shape}")

    # Compare with PyTorch
    with torch.no_grad():
        torch_out = wrapper(input_ids, attention_mask, token_type_ids)
    print(f"PyTorch output shape: {torch_out.shape}")
    diff = np.abs(onnx_out[0] - torch_out.numpy()).max()
    print(f"Max diff: {diff}")


if __name__ == "__main__":
    export_bert_onnx()
