import torch

state_dict = torch.load('PEFT_model.pt')
print(sum([p.numel() for n, p in state_dict.items()]))

# state_dict = torch.load("hw3_data/p2_data/decoder_model.bin")
# fix_param = [key for key in state_dict]

# model_state = torch.load("model_3.pt")
# param_store = {}
# for name, param in model_state.items():
#     if any(name.endswith(key) for key in fix_param):
#         print(name)
#     else:
#         param_store[name] = param

# torch.save(param_store, "test.pt")