import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from find_torch_api import find_torch_imports, find_torch_usage, load_tensor_methods  # noqa: E402


TENSOR_METHODS = load_tensor_methods()


def scan_apis(content):
    return {
        item["api"]
        for item in find_torch_usage(
            content,
            find_torch_imports(content),
            mode="static",
            tensor_methods=TENSOR_METHODS,
        )
    }


class FindTorchApiStaticInferenceTest(unittest.TestCase):
    def assert_detects(self, content, *apis):
        detected = scan_apis(content)
        missing = set(apis) - detected
        self.assertFalse(missing, f"missing {sorted(missing)} from {sorted(detected)}")

    def test_tensor_from_torch_factory_keeps_subscript_item(self):
        self.assert_detects(
            """
import torch
a = torch.tensor([1, 2, 3])
b = a[0].item()
""",
            "torch.tensor",
            "torch.Tensor.item",
        )

    def test_parameter_data_from_named_parameters(self):
        self.assert_detects(
            """
for name, param in model.named_parameters():
    a = param.data
""",
            "torch.Tensor.data",
        )

    def test_parameter_data_copy_chain(self):
        self.assert_detects(
            """
for name, param in model.named_parameters():
    param.data.copy_(x)
""",
            "torch.Tensor.copy_",
        )

    def test_parameter_data_flows_into_function_param(self):
        self.assert_detects(
            """
def outer(param):
    xx(param.data)

def xx(t):
    input_shape = t.size()
    attr_shape = t.size
""",
            "torch.Tensor.size",
        )

    def test_function_tensor_param_chain_assignment(self):
        self.assert_detects(
            """
def outer(param):
    xx(param.data)

def xx(t):
    t = t.transpose(0, 1).contiguous()
""",
            "torch.Tensor.transpose",
            "torch.Tensor.contiguous",
        )

    def test_full_tensor_chain_and_variable_reuse(self):
        self.assert_detects(
            """
full_tensor = param.full_tensor().detach()
full_tensor = full_tensor.cpu()
shape = full_tensor.shape
""",
            "torch.Tensor.detach",
            "torch.Tensor.cpu",
            "torch.Tensor.shape",
        )

    def test_huggingface_training_step_tensor_chains(self):
        self.assert_detects(
            """
import torch
import torch.nn.functional as F

def train_step(batch, device, vocab_size):
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.ones_like(input_ids).float()
    labels = torch.tensor(batch["labels"]).view(-1)
    logits = torch.randn(input_ids.size(0), input_ids.size(1), vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels)
    loss.backward()
    return loss.item()
""",
            "torch.tensor",
            "torch.ones_like",
            "torch.randn",
            "torch.nn.functional.cross_entropy",
            "torch.Tensor.to",
            "torch.Tensor.float",
            "torch.Tensor.view",
            "torch.Tensor.size",
            "torch.Tensor.backward",
            "torch.Tensor.item",
        )

    def test_huggingface_no_grad_inference_chains(self):
        self.assert_detects(
            """
import torch

@torch.no_grad()
def infer(batch, device):
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    logits = torch.randn(input_ids.size(0), 10)
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    return pred
""",
            "torch.no_grad",
            "torch.tensor",
            "torch.randn",
            "torch.Tensor.to",
            "torch.Tensor.size",
            "torch.Tensor.argmax",
            "torch.Tensor.detach",
            "torch.Tensor.cpu",
            "torch.Tensor.numpy",
        )

    def test_huggingface_generation_loop_chains(self):
        self.assert_detects(
            """
import torch

def generate(prompt_ids, vocab_size):
    input_ids = torch.tensor(prompt_ids).unsqueeze(0).to("cuda")
    for _ in range(2):
        logits = torch.randn(input_ids.size(0), input_ids.size(1), vocab_size)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids.squeeze(0).tolist()
""",
            "torch.tensor",
            "torch.randn",
            "torch.argmax",
            "torch.cat",
            "torch.Tensor.unsqueeze",
            "torch.Tensor.to",
            "torch.Tensor.size",
            "torch.Tensor.squeeze",
            "torch.Tensor.tolist",
        )


if __name__ == "__main__":
    unittest.main()
