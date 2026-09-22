import torch
import requests

response = requests.get("https://www.google.com")
print(response)

msg = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(msg)
# Example tensor operation
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print("Sum:", a + b)
