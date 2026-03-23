import re
import matplotlib.pyplot as plt
import os


def parse_log_file(filename):
    """
    解析日志文件，提取每一轮的训练和测试指标
    """
    metrics = {
        'train_acc': [],
        'train_loss': [],
        'test_acc': [],
        'test_loss': []
    }

    if not os.path.exists(filename):
        print(f"找不到文件: {filename}")
        return metrics

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if 'train_acc:' in line and 'test_acc:' in line:
                matches = re.findall(r'(train_acc|train_loss|test_acc|test_loss):\s*([0-9\.]+)', line)
                for key, value in matches:
                    metrics[key].append(float(value))

    return metrics


if __name__ == '__main__':
    # ================= 配置区 =================
    # 基线日志文件路径 (全程 FedAvg, 40%噪声)
    baseline_file = 'results100_mnist_fedrn_20260320-171728.txt'
    # 改进方法日志文件路径 (FedER, 40%噪声)
    feder_file = 'results_Hierarchical_mnist_fedrn_20260320-152257.txt'

    # 输出的图片名称
    output_image = 'feder_vs_fedavg_noise40.png'
    # ==========================================

    baseline_metrics = parse_log_file(baseline_file)
    feder_metrics = parse_log_file(feder_file)

    if not baseline_metrics['train_acc'] or not feder_metrics['train_acc']:
        print("解析失败，请检查文件名或文件内容是否正确！")
        exit()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    plot_config = [
        ('train_acc', 'Train Accuracy (%)', axs[0, 0]),
        ('test_acc', 'Test Accuracy (%)', axs[0, 1]),
        ('train_loss', 'Train Loss', axs[1, 0]),
        ('test_loss', 'Test Loss', axs[1, 1])
    ]

    color_baseline = '#1f77b4'  # FedAvg - 蓝色
    color_feder = '#ff7f0e'  # FedER - 橙色

    for key, title, ax in plot_config:
        ax.plot(baseline_metrics[key], label='FedAvg (Baseline)',
                color=color_baseline, linewidth=2, linestyle='--')
        ax.plot(feder_metrics[key], label='FedER (Improved)',
                color=color_feder, linewidth=2, linestyle='-')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel(title.split(' ')[0] + ' Value', fontsize=12)

        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"🎉 40% 噪声绘图完成！图片已保存为: {output_image}")