import torch

# dictionary 출력시 보기 좋게 해주는 function
def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + '【'+ str(key) + '】')
            print_dict(value, indent+1)
        else:
            print('\t' * indent + '【'+ str(key) + '】', end=' : ')
            print(str(value))


# Recursive tensor to list
def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()  # 텐서를 리스트로 변환
    elif isinstance(data, dict):
        return {key: tensor_to_list(value) for key, value in data.items()}  # 딕셔너리도 재귀적으로 처리
    else:
        return data

# Recusive list to tensor
def list_to_tensor(data):
    if isinstance(data, list):
        return torch.tensor(data)  # 리스트를 텐서로 변환
    elif isinstance(data, dict):  # 딕셔너리 내부에도 리스트가 있을 수 있으므로 재귀 처리
        return {key: list_to_tensor(value) for key, value in data.items()}
    else:
        return data
