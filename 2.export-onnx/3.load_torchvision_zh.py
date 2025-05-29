import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import os
import argparse
import torchvision  # 加载预训练视觉模型所需的依赖

def get_model(type, dir):
    """
    从torchvision选择预训练视觉模型并定义其ONNX输出文件路径。

    参数:
        type (str): 模型类型（如'resnet', 'vgg', 'mobilenet'等）。
        dir (str): 保存ONNX模型的目录。

    返回:
        tuple: (model, file_path)，其中model为PyTorch模型，file_path为ONNX文件路径。
    """
    if type == "resnet":
        model = torchvision.models.resnet50()
        file = os.path.join(dir, "resnet50.onnx")
    elif type == "vgg":
        model = torchvision.models.vgg11()
        file = os.path.join(dir, "vgg11.onnx")
    elif type == "mobilenet":
        model = torchvision.models.mobilenet_v3_small()
        file = os.path.join(dir, "mobilenet_v3_small.onnx")
    elif type == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        file = os.path.join(dir, "efficientnet_b0.onnx")
    elif type == "efficientnetv2":
        model = torchvision.models.efficientnet_v2_s()
        file = os.path.join(dir, "efficientnet_v2_s.onnx")
    elif type == "regnet":
        model = torchvision.models.regnet_x_1_6gf()
        file = os.path.join(dir, "regnet_x_1_6gf.onnx")
    else:
        raise ValueError(f"不支持的模型类型: {type}")
    return model, file

def infer(model, input_tensor):
    """使用指定模型和输入张量进行推理。

    参数:
        model (nn.Module): 来自torchvision的预训练PyTorch模型。
        input_tensor (torch.Tensor): 输入张量，形状为 (batch_size, 3, 224, 224)。

    返回:
        torch.Tensor: 模型输出张量。
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算以进行推理
        output = model(input_tensor)
    print("推理输出形状:", output.shape)
    return output

def export_norm_onnx(model, file, input_tensor):
    """将模型导出为ONNX格式，进行验证和简化。

    参数:
        model (nn.Module): 要导出的PyTorch模型。
        file (str): 保存ONNX模型的路径。
        input_tensor (torch.Tensor): 用于跟踪的虚拟输入张量，形状为 (batch_size, 3, 224, 224)。

    注意:
        - do_constant_folding=True可能将BatchNorm融合到Conv层，减少节点数。
        - ONNX计算图可能包含Identity节点，源于残差连接（如ResNet）或未完全优化的操作。
        - 使用onnx-simplifier可减少冗余节点（如Identity）。
        - 可用Netron检查导出的模型，分析Identity节点来源。
    """
    # 确保模型处于评估模式
    model.eval()

    # 导出模型到ONNX格式
    torch.onnx.export(
        model=model,              # 要导出的模型
        args=(input_tensor,),     # 虚拟输入，用于跟踪计算图
        f=file,                   # 输出文件路径
        input_names=["input0"],   # 输入张量名称
        output_names=["output0"], # 输出张量名称
        opset_version=15,         # ONNX操作集版本，确保兼容性
        dynamic_axes={
            'input0': {0: 'batch'},   # 支持动态批次大小
            'output0': {0: 'batch'}   # 支持动态批次大小
        },
        do_constant_folding=True, # 优化：折叠常量（如BatchNorm到Conv）
        verbose=True              # 打印导出详情，便于调试
    )
    print(f"完成普通ONNX导出，保存至: {file}")

    # 加载并验证ONNX模型
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)  # 检查模型完整性

    # 使用onnx-simplifier简化ONNX模型
    # 这会移除冗余节点（如残差连接中的Identity节点）
    print(f"使用onnx-simplifier {onnxsim.__version__} 进行简化...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "ONNX简化失败"
    onnx.save(model_onnx, file)  # 保存简化后的模型
    print(f"完成ONNX简化，模型保存至: {file}")

def main(args):
    """主函数，选择并导出预训练视觉模型到ONNX。

    参数:
        args: 包含模型类型和输出目录的命令行参数。

    注意:
        - 需要torchvision提供预训练模型。通过requirements.txt安装依赖（如torch==2.7.0, torchvision）。
        - 导出的ONNX计算图可能包含Identity节点，源于残差连接或BatchNorm融合。使用Netron检查计算图。
    """
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，此脚本需要GPU。")

    # 创建输出目录
    os.makedirs(args.dir, exist_ok=True)  # 确保目录存在

    # 定义输入张量（batch_size=1, 通道=3, 高=224, 宽=224）
    input_tensor = torch.rand(1, 3, 224, 224, device='cuda')

    # 获取模型和输出文件路径
    model, file = get_model(args.type, args.dir)

    # 将模型移动到CUDA
    model.cuda()

    # 执行推理
    infer(model, input_tensor)

    # 导出模型到ONNX
    export_norm_onnx(model, file, input_tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将预训练视觉模型导出为ONNX格式")
    parser.add_argument("-t", "--type", type=str, default="resnet",
                        help="模型类型：resnet, vgg, mobilenet, efficientnet, efficientnetv2, regnet")
    parser.add_argument("-d", "--dir", type=str, default="./2.export-onnx/models/",
                        help="保存ONNX模型的目录")
    
    opt = parser.parse_args()
    main(opt)
